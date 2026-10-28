// SPDX-License-Identifier: Apache-2.0
/*
 * engram_nvme.c -- node-local NVMe row reader for the DeepSeek-V4.1 Engram tables (recipe patch 016).
 *
 * WHAT: a native thread (no GIL, no CUDA calls) that serves "row requests" for one or more Engram layers. A channel is
 * one layer's table: the fp8 value rows (256 B) and e8m0 scale rows (8 B) live in two tensors of a safetensors shard.
 * The GPU publishes a request into pinned host memory (ids + a doorbell, `req_seq`); this thread reads the rows with
 * O_DIRECT, compacts them into a pinned staging buffer in the order the dequant kernel consumes them, then publishes
 * `done_seq`. The GPU waits on `done_seq` inside the CUDA graph (see engram_nvme.py).
 *
 * I/O PATHS (no liburing, no libaio, no io_uring: raw syscalls only, all allowed by Docker's default seccomp profile):
 *   - Linux native AIO (io_setup/io_submit/io_getevents) with O_DIRECT          [primary]
 *   - blocking pread(O_DIRECT) from a small thread pool                          [fallback]
 *
 * FAIL CLOSED: an I/O error, a short read, a protocol violation, a request pending longer than `timeout_ms`, a GPU-side
 * wait timeout (`gpu_abort`) or a changed shard file marks `status` != 0 in every channel (the dequant kernel then writes
 * NaN rows) and ends the process with _exit(86) so the executor tears the engine down. Rows are never guessed.
 *
 * ROW IDS: ids outside the rank's slice [row_lo, row_hi) read as zero rows, exactly like the resident kernel's `owned` mask
 * (values 0, scale byte 0 -> 0.0 after dequantization); they are counted in stats.oob_rows.
 *
 * Build: gcc -O2 -shared -fPIC -pthread -o libengram_nvme.so engram_nvme.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define EN_ABI_VERSION 1
#define EN_ROW_V 256u
#define EN_ROW_S 8u
#define EN_MAX_CHAN 8
#define EN_STALL_N 512
#define EN_EXIT_CODE 86
#define EN_WINSLOT 8192u /* window buffer per in-flight read: enough for 4 KiB sector devices */
#define EN_SUBMIT_MAX 64u
#define EN_EVENTS_MAX 128u
#define EN_HIST 24

enum { EN_BACKEND_AUTO = 0, EN_BACKEND_AIO = 1, EN_BACKEND_THREADS = 2 };
enum { EN_POLL_AUTO = 0, EN_POLL_SPIN = 1, EN_POLL_SLEEP = 2 };
enum { EN_FLAG_DEDUPE = 1, EN_FLAG_NO_ABORT = 2 };
enum {
    EN_OK = 0,
    EN_ERR_IO = 1,
    EN_ERR_SHORT = 2,
    EN_ERR_PROTO = 3,
    EN_ERR_TIMEOUT = 4,
    EN_ERR_GPU_ABORT = 5,
    EN_ERR_VERIFY = 6,
    EN_ERR_FILE_CHANGED = 7,
    EN_ERR_INTERNAL = 8
};

/* ---- control page shared with the GPU and Python (pinned host memory, 8 KiB) -------------------------------------- */
typedef struct en_ctrl {
    /* 0x000 */ volatile uint64_t req_seq; /* GPU -> CPU doorbell: st.release.sys of the new sequence number */
    uint8_t _p0[56];
    /* 0x040 */ volatile uint32_t req_ntok; /* GPU -> CPU: tokens in the request (written before req_seq) */
    uint8_t _p1[60];
    /* 0x080 */ volatile uint64_t done_seq; /* CPU -> GPU: rows of request `done_seq` are staged (release) */
    uint8_t _p2[56];
    /* 0x0C0 */ volatile uint32_t status; /* 0 ok; EN_ERR_* once the channel failed (sticky) */
    volatile uint32_t gpu_abort;          /* GPU -> CPU: the wait kernel gave up / saw a stale request */
    volatile uint32_t err_detail;
    uint8_t _p3[52];
    /* 0x100 */ volatile uint64_t heartbeat_ns;
    volatile uint64_t served; /* requests completed */
    volatile uint64_t last_ns;
    uint8_t _p4[40];
    uint8_t _pad[4096 - 0x140];
    /* 0x1000 */ volatile uint64_t stall[EN_STALL_N]; /* GPU -> CPU: (seq << 32) | wait-kernel polls, index seq % 512 */
} en_ctrl;

_Static_assert(offsetof(en_ctrl, req_seq) == 0x000, "req_seq");
_Static_assert(offsetof(en_ctrl, req_ntok) == 0x040, "req_ntok");
_Static_assert(offsetof(en_ctrl, done_seq) == 0x080, "done_seq");
_Static_assert(offsetof(en_ctrl, status) == 0x0C0, "status");
_Static_assert(offsetof(en_ctrl, gpu_abort) == 0x0C4, "gpu_abort");
_Static_assert(offsetof(en_ctrl, heartbeat_ns) == 0x100, "heartbeat");
_Static_assert(offsetof(en_ctrl, stall) == 0x1000, "stall");
_Static_assert(sizeof(en_ctrl) == 0x2000, "en_ctrl size");

typedef struct en_cfg {
    int32_t abi;          /* must be EN_ABI_VERSION */
    int32_t backend;      /* EN_BACKEND_* */
    int32_t queue_depth;  /* AIO: reads in flight (<= 4096) */
    int32_t nthreads;     /* thread-pool size (fallback) */
    int32_t poll;         /* EN_POLL_* */
    int32_t spin_us;      /* auto: spin this long after the last activity, then sleep */
    int32_t sleep_us;     /* sleep step when idle */
    int32_t keepwarm_us;  /* heartbeat read period, 0 = off */
    int32_t keepwarm_idle_ms; /* stop the heartbeat after this long without a request */
    int32_t timeout_ms;   /* a request pending longer than this is fatal */
    int32_t verify_pct;   /* re-read this % of delivered rows through a buffered pread and compare */
    int32_t cpu;          /* reader thread affinity, -1 = none */
    int32_t flags;        /* EN_FLAG_* */
    int32_t fault_every;  /* test: every Nth read fails (0 = off) */
    int32_t fault_kind;   /* test: 1 = EIO, 2 = short read */
    int32_t delay_us;     /* test: sleep this long before serving each request */
    int32_t hang_at;      /* test: the reader hangs forever at this request number (0 = off) */
    int32_t max_rows;     /* largest cap_rows of any channel: the request scratch is sized for it up front */
    int32_t _pad;
} en_cfg;

typedef struct en_chan_spec {
    const char *path;
    uint64_t values_off; /* absolute file offset of row 0 of the value tensor */
    uint64_t scales_off; /* absolute file offset of row 0 of the scale tensor */
    uint64_t num_rows;
    uint64_t row_lo, row_hi; /* rows owned by this rank: [lo, hi) */
    en_ctrl *ctrl;
    int32_t *req_idx; /* [2][cap_rows] global row ids, filled by the GPU */
    uint8_t *stage_v; /* [2][cap_rows][256] */
    uint8_t *stage_s; /* [2][cap_rows][8] */
    uint32_t cap_rows;     /* rows per slot (= max tokens * local_heads) */
    uint32_t local_heads;  /* ids per token */
} en_chan_spec;

typedef struct en_stats {
    uint64_t requests, rows, unique_rows, oob_rows, dup_rows, reads, read_bytes, keepwarm_reads, verify_rows, errors;
    uint64_t lat_hist[EN_HIST]; /* request latency, doorbell seen -> done_seq published, log2(us) buckets */
    uint64_t lat_sum_us, lat_max_us;
    uint64_t stall_hist[EN_HIST]; /* GPU wait-kernel polls per wait, log2 buckets */
    uint64_t stall_n;
    int32_t backend;
    int32_t nchan;
    uint32_t qd_max;
    uint32_t align;
} en_stats;

/* ---- internals -------------------------------------------------------------------------------------------------- */
typedef struct en_job {
    uint64_t off;     /* aligned file offset */
    uint32_t len;     /* aligned length */
    uint32_t win_off; /* wanted bytes start here inside the window */
    uint32_t copy;    /* wanted bytes: 256 or 8 */
    uint32_t _pad;
    uint8_t *dst;
} en_job;

typedef struct en_chan {
    int fd, fd_buf;
    uint64_t values_off, scales_off, num_rows, row_lo, row_hi;
    en_ctrl *ctrl;
    int32_t *req_idx;
    uint8_t *stage_v, *stage_s;
    uint32_t cap_rows, local_heads;
    uint64_t last_seq;
    uint64_t stall_seen;
    /* identity of the shard, rechecked by the watchdog */
    uint64_t st_ino, st_size;
    int64_t st_mtime_ns;
    /* watchdog */
    uint64_t pend_since_ns;
    uint64_t pend_seq;
    char path[512];
} en_chan;

typedef struct en_pool en_pool;
typedef struct en_handle en_handle;

struct en_pool {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t gen;
    en_job *jobs;
    uint32_t nj;
    int fd;
    uint32_t next, done;
    int err;
    int stop;
    int nworkers;
    pthread_t *th;
    uint8_t *bufs; /* nworkers + 1 windows (the reader participates) */
    en_handle *h;
};

struct en_handle {
    en_cfg cfg;
    int backend;
    uint32_t align; /* O_DIRECT offset/length alignment */
    en_chan chans[EN_MAX_CHAN];
    int nchan;
    int stop;
    int started;
    int failed; /* a fatal error was already reported */
    pthread_t reader, keepwarm, watchdog;
    /* per-request scratch (sized by the largest channel) */
    en_job *jobs;
    uint32_t jobs_cap;
    uint32_t *ht_key, *ht_val;
    uint32_t ht_cap;
    uint32_t *dup_dst, *dup_src;
    uint32_t dup_cap;
    /* AIO */
    aio_context_t ctx;
    int aio_ready;
    uint32_t qd;
    struct iocb *iocbs;
    struct iocb **iocb_ptrs;
    struct io_event *events;
    uint8_t *win;
    uint32_t *free_slots;
    uint32_t nfree;
    /* thread pool */
    en_pool pool;
    /* activity, shared with the keep-warm thread */
    uint64_t last_activity_ns;
    /* test hooks */
    uint64_t nreads;
    uint64_t nreq;
    en_stats st;
    pthread_mutex_t st_mu;
};

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__)
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

/* full system-wide barrier before a flag the GPU polls: belt and braces on top of the release store */
static inline void sys_barrier(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static void en_log(const char *fmt, ...) {
    char buf[512];
    int n = snprintf(buf, sizeof buf, "[engram-nvme] ");
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(buf + n, sizeof buf - (size_t)n - 2, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 2) n = (int)sizeof buf - 2;
    buf[n++] = '\n';
    if (write(2, buf, (size_t)n) < 0) { /* nothing to do */ }
}

static void set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static inline int log2_bucket(uint64_t v) {
    int b = 0;
    while (v > 1 && b < EN_HIST - 1) {
        v >>= 1;
        b++;
    }
    return b;
}

/* ---- raw AIO syscalls --------------------------------------------------------------------------------------------- */
static inline long sys_io_setup(unsigned nr, aio_context_t *ctx) { return syscall(SYS_io_setup, nr, ctx); }
static inline long sys_io_destroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }
static inline long sys_io_submit(aio_context_t ctx, long n, struct iocb **p) { return syscall(SYS_io_submit, ctx, n, p); }
static inline long sys_io_getevents(aio_context_t ctx, long min_nr, long nr, struct io_event *ev, struct timespec *to) {
    return syscall(SYS_io_getevents, ctx, min_nr, nr, ev, to);
}

/* ---- fatal ------------------------------------------------------------------------------------------------------- */
static void en_fatal(en_handle *h, int code, const char *fmt, ...) {
    char msg[300];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (__atomic_exchange_n(&h->failed, 1, __ATOMIC_ACQ_REL)) return; /* the first error is the one that matters */
    int nchan = __atomic_load_n(&h->nchan, __ATOMIC_ACQUIRE);
    for (int i = 0; i < nchan; i++) {
        uint32_t zero = 0;
        __atomic_compare_exchange_n(&h->chans[i].ctrl->status, &zero, (uint32_t)code, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    sys_barrier();
    en_log("FATAL (code %d): %s", code, msg);
    __atomic_add_fetch(&h->st.errors, 1, __ATOMIC_RELAXED);
    if (h->cfg.flags & EN_FLAG_NO_ABORT) return;
    usleep(200000); /* let the log line out and the GPU see the NaN rows */
    _exit(EN_EXIT_CODE);
}

/* ---- one job's completion: copy the wanted bytes out of the aligned window ---------------------------------------- */
static inline int job_check_copy(const en_job *j, const uint8_t *win, int64_t res) {
    if (res < 0) return EN_ERR_IO;
    if ((uint64_t)res < (uint64_t)j->win_off + j->copy) return EN_ERR_SHORT;
    memcpy(j->dst, win + j->win_off, j->copy);
    return EN_OK;
}

/* ---- AIO runner --------------------------------------------------------------------------------------------------- */
static int run_jobs_aio(en_handle *h, en_chan *c, en_job *jobs, uint32_t nj) {
    uint32_t next = 0, inflight = 0, done = 0;
    uint64_t t_start = now_ns();
    uint64_t timeout_ns = (uint64_t)h->cfg.timeout_ms * 1000000ull;
    int rc = EN_OK;
    while (done < nj) {
        uint32_t nsub = 0;
        while (rc == EN_OK && next < nj && inflight < h->qd && h->nfree > 0 && nsub < EN_SUBMIT_MAX) {
            uint32_t s = h->free_slots[--h->nfree];
            en_job *j = &jobs[next];
            struct iocb *cb = &h->iocbs[s];
            memset(cb, 0, sizeof *cb);
            cb->aio_data = ((uint64_t)next << 32) | s;
            cb->aio_lio_opcode = IOCB_CMD_PREAD;
            cb->aio_fildes = (uint32_t)c->fd;
            cb->aio_buf = (uint64_t)(uintptr_t)(h->win + (size_t)s * EN_WINSLOT);
            cb->aio_nbytes = j->len;
            cb->aio_offset = (int64_t)j->off;
            h->iocb_ptrs[nsub++] = cb;
            next++;
            inflight++;
        }
        if (nsub) {
            uint32_t sent = 0;
            while (sent < nsub) {
                long r = sys_io_submit(h->ctx, (long)(nsub - sent), h->iocb_ptrs + sent);
                if (r < 0) {
                    if (errno == EINTR) continue;
                    if (errno == EAGAIN) break; /* queue full: roll the tail back, reap, retry */
                    return EN_ERR_IO;
                }
                sent += (uint32_t)r;
                if (r == 0) break;
            }
            if (sent < nsub) { /* roll back the iocbs that were prepared but not accepted (they are the last ones) */
                for (uint32_t k = sent; k < nsub; k++) {
                    uint32_t s = (uint32_t)(h->iocb_ptrs[k]->aio_data & 0xffffffffu);
                    h->free_slots[h->nfree++] = s;
                }
                next -= nsub - sent;
                inflight -= nsub - sent;
                if (inflight == 0) {
                    if (now_ns() - t_start > timeout_ns) return EN_ERR_TIMEOUT;
                    usleep(100);
                }
            }
        }
        if (inflight) {
            int can_submit = (rc == EN_OK && next < nj && h->nfree > 0 && inflight < h->qd);
            long min_nr = can_submit ? 0 : 1;
            struct timespec ts = {0, can_submit ? 0 : 20 * 1000 * 1000};
            long ne = sys_io_getevents(h->ctx, min_nr, EN_EVENTS_MAX, h->events, &ts);
            if (ne < 0) {
                if (errno == EINTR) continue;
                return EN_ERR_IO;
            }
            for (long k = 0; k < ne; k++) {
                struct io_event *e = &h->events[k];
                uint32_t jidx = (uint32_t)(e->data >> 32), s = (uint32_t)(e->data & 0xffffffffu);
                int64_t res = (int64_t)e->res;
                en_job *j = &jobs[jidx];
                uint64_t n = ++h->nreads;
                if (h->cfg.fault_every > 0 && (n % (uint64_t)h->cfg.fault_every) == 0)
                    res = (h->cfg.fault_kind == 2) ? 1 : -EIO;
                int r2 = job_check_copy(j, h->win + (size_t)s * EN_WINSLOT, res);
                if (r2 != EN_OK && rc == EN_OK) rc = r2;
                h->free_slots[h->nfree++] = s;
                inflight--;
                done++;
            }
            if (ne == 0 && !can_submit && now_ns() - t_start > timeout_ns) return EN_ERR_TIMEOUT;
            if (rc != EN_OK && inflight == 0) return rc;
        }
    }
    return rc;
}

/* ---- thread-pool runner (fallback) ---------------------------------------------------------------------------------- */
static int pool_do_job(en_pool *p, en_job *j, uint8_t *win) {
    ssize_t r;
    do {
        r = pread(p->fd, win, j->len, (off_t)j->off);
    } while (r < 0 && errno == EINTR);
    en_handle *h = p->h;
    uint64_t n = __atomic_add_fetch(&h->nreads, 1, __ATOMIC_RELAXED);
    if (h->cfg.fault_every > 0 && (n % (uint64_t)h->cfg.fault_every) == 0) r = (h->cfg.fault_kind == 2) ? 1 : -1;
    return job_check_copy(j, win, (int64_t)r);
}

static void pool_drain(en_pool *p, uint8_t *win) {
    uint32_t i;
    while ((i = __atomic_fetch_add(&p->next, 1, __ATOMIC_ACQUIRE)) < __atomic_load_n(&p->nj, __ATOMIC_RELAXED)) {
        int r = pool_do_job(p, &p->jobs[i], win);
        if (r != EN_OK) {
            int zero = 0;
            __atomic_compare_exchange_n(&p->err, &zero, r, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        }
        __atomic_add_fetch(&p->done, 1, __ATOMIC_RELEASE);
    }
}

static void *pool_worker(void *arg) {
    void **a = arg;
    en_pool *p = a[0];
    int idx = (int)(intptr_t)a[1];
    free(a);
    uint8_t *win = p->bufs + (size_t)idx * EN_WINSLOT;
    uint64_t my_gen = 0;
    prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->stop && p->gen == my_gen) pthread_cond_wait(&p->cv, &p->mu);
        if (p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        my_gen = p->gen;
        pthread_mutex_unlock(&p->mu);
        pool_drain(p, win);
    }
}

static int run_jobs_threads(en_handle *h, en_chan *c, en_job *jobs, uint32_t nj) {
    en_pool *p = &h->pool;
    pthread_mutex_lock(&p->mu);
    p->jobs = jobs;
    __atomic_store_n(&p->nj, nj, __ATOMIC_RELAXED);
    p->fd = c->fd;
    __atomic_store_n(&p->err, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&p->done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&p->next, 0, __ATOMIC_RELEASE); /* last: workers only take jobs once everything above is visible */
    p->gen++;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    pool_drain(p, p->bufs + (size_t)p->nworkers * EN_WINSLOT); /* the reader works too */
    uint64_t t_start = now_ns(), timeout_ns = (uint64_t)h->cfg.timeout_ms * 1000000ull;
    while (__atomic_load_n(&p->done, __ATOMIC_ACQUIRE) < nj) {
        sched_yield();
        if (now_ns() - t_start > timeout_ns) return EN_ERR_TIMEOUT;
    }
    return __atomic_load_n(&p->err, __ATOMIC_RELAXED);
}

/* ---- request ---------------------------------------------------------------------------------------------------- */
static inline uint32_t ht_hash(uint32_t k, uint32_t mask) { return (k * 2654435761u) >> 7 & mask; }

static int verify_rows(en_handle *h, en_chan *c, const int32_t *ids, uint32_t n, const uint8_t *sv, const uint8_t *ss, uint64_t seq) {
    if (h->cfg.verify_pct <= 0 || c->fd_buf < 0) return EN_OK;
    uint8_t tmp[EN_ROW_V];
    uint64_t rng = seq * 0x9E3779B97F4A7C15ull + 12345;
    for (uint32_t i = 0; i < n; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        if ((int)(rng % 100) >= h->cfg.verify_pct) continue;
        uint64_t id = (uint64_t)(uint32_t)ids[i];
        if (id < c->row_lo || id >= c->row_hi) continue;
        off_t ov = (off_t)(c->values_off + id * EN_ROW_V), os = (off_t)(c->scales_off + id * EN_ROW_S);
        if (pread(c->fd_buf, tmp, EN_ROW_V, ov) != (ssize_t)EN_ROW_V || memcmp(tmp, sv + (size_t)i * EN_ROW_V, EN_ROW_V) != 0) return EN_ERR_VERIFY;
        if (pread(c->fd_buf, tmp, EN_ROW_S, os) != (ssize_t)EN_ROW_S || memcmp(tmp, ss + (size_t)i * EN_ROW_S, EN_ROW_S) != 0) return EN_ERR_VERIFY;
        posix_fadvise(c->fd_buf, ov & ~4095ll, 4096, POSIX_FADV_DONTNEED);
        posix_fadvise(c->fd_buf, os & ~4095ll, 4096, POSIX_FADV_DONTNEED);
        h->st.verify_rows++;
    }
    return EN_OK;
}

static int serve_request(en_handle *h, en_chan *c, uint64_t seq, uint64_t t_seen) {
    en_ctrl *ct = c->ctrl;
    uint32_t ntok = ct->req_ntok;
    uint32_t n = ntok * c->local_heads;
    (void)t_seen;
    if (ntok == 0 || n > c->cap_rows) {
        en_fatal(h, EN_ERR_PROTO, "request %llu: %u tokens do not fit (capacity %u rows)", (unsigned long long)seq, ntok, c->cap_rows);
        return EN_ERR_PROTO;
    }
    size_t slot = (size_t)(seq & 1);
    const int32_t *ids = c->req_idx + slot * c->cap_rows;
    uint8_t *sv = c->stage_v + slot * (size_t)c->cap_rows * EN_ROW_V;
    uint8_t *ss = c->stage_s + slot * (size_t)c->cap_rows * EN_ROW_S;
    const uint32_t A = h->align;
    uint32_t nj = 0, nd = 0, oob = 0;
    int dedupe = (h->cfg.flags & EN_FLAG_DEDUPE) != 0;
    uint32_t mask = 0;
    if (dedupe) {
        uint32_t cap = 1;
        while (cap < 2 * n) cap <<= 1;
        mask = cap - 1;
        memset(h->ht_key, 0, (size_t)cap * sizeof(uint32_t));
    }
    for (uint32_t i = 0; i < n; i++) {
        uint64_t id = (uint64_t)(uint32_t)ids[i];
        if (id < c->row_lo || id >= c->row_hi) {
            memset(sv + (size_t)i * EN_ROW_V, 0, EN_ROW_V);
            memset(ss + (size_t)i * EN_ROW_S, 0, EN_ROW_S);
            oob++;
            continue;
        }
        if (dedupe) {
            uint32_t key = (uint32_t)id + 1u, pos = ht_hash(key, mask);
            int found = 0;
            while (h->ht_key[pos]) {
                if (h->ht_key[pos] == key) {
                    h->dup_dst[nd] = i;
                    h->dup_src[nd] = h->ht_val[pos];
                    nd++;
                    found = 1;
                    break;
                }
                pos = (pos + 1) & mask;
            }
            if (found) continue;
            h->ht_key[pos] = key;
            h->ht_val[pos] = i;
        }
        uint64_t ov = c->values_off + id * EN_ROW_V, os = c->scales_off + id * EN_ROW_S;
        en_job *jv = &h->jobs[nj++], *js = &h->jobs[nj++];
        jv->off = ov & ~(uint64_t)(A - 1);
        jv->win_off = (uint32_t)(ov - jv->off);
        jv->copy = EN_ROW_V;
        jv->len = (jv->win_off + EN_ROW_V + A - 1) & ~(A - 1);
        jv->dst = sv + (size_t)i * EN_ROW_V;
        js->off = os & ~(uint64_t)(A - 1);
        js->win_off = (uint32_t)(os - js->off);
        js->copy = EN_ROW_S;
        js->len = (js->win_off + EN_ROW_S + A - 1) & ~(A - 1);
        js->dst = ss + (size_t)i * EN_ROW_S;
    }
    h->st.requests++;
    h->st.rows += n;
    h->st.oob_rows += oob;
    h->st.dup_rows += nd;
    h->st.unique_rows += nj / 2;
    h->st.reads += nj;
    for (uint32_t k = 0; k < nj; k++) h->st.read_bytes += h->jobs[k].len;
    if (nj > h->st.qd_max) h->st.qd_max = nj < h->qd ? nj : h->qd;
    int rc = EN_OK;
    if (nj) rc = (h->backend == EN_BACKEND_AIO) ? run_jobs_aio(h, c, h->jobs, nj) : run_jobs_threads(h, c, h->jobs, nj);
    if (rc != EN_OK) {
        en_fatal(h, rc == EN_ERR_SHORT ? EN_ERR_SHORT : (rc == EN_ERR_TIMEOUT ? EN_ERR_TIMEOUT : EN_ERR_IO),
                 "request %llu (%u rows, %u reads, %s): read failed (%s)", (unsigned long long)seq, n, nj,
                 h->backend == EN_BACKEND_AIO ? "aio" : "threads",
                 rc == EN_ERR_SHORT ? "short read" : (rc == EN_ERR_TIMEOUT ? "timeout" : "I/O error"));
        return rc;
    }
    for (uint32_t k = 0; k < nd; k++) {
        memcpy(sv + (size_t)h->dup_dst[k] * EN_ROW_V, sv + (size_t)h->dup_src[k] * EN_ROW_V, EN_ROW_V);
        memcpy(ss + (size_t)h->dup_dst[k] * EN_ROW_S, ss + (size_t)h->dup_src[k] * EN_ROW_S, EN_ROW_S);
    }
    rc = verify_rows(h, c, ids, n, sv, ss, seq);
    if (rc != EN_OK) {
        en_fatal(h, EN_ERR_VERIFY, "request %llu: a delivered row differs from the buffered re-read", (unsigned long long)seq);
        return rc;
    }
    return EN_OK;
}

/* ---- threads ---------------------------------------------------------------------------------------------------- */
static void pin_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void *reader_main(void *arg) {
    en_handle *h = arg;
    prctl(PR_SET_NAME, "engram-nvme-rd", 0, 0, 0);
    prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);
    pin_cpu(h->cfg.cpu);
    uint64_t last_active = now_ns();
    while (!__atomic_load_n(&h->stop, __ATOMIC_ACQUIRE)) {
        int nchan = __atomic_load_n(&h->nchan, __ATOMIC_ACQUIRE);
        int did = 0;
        uint64_t t = now_ns();
        for (int i = 0; i < nchan; i++) {
            en_chan *c = &h->chans[i];
            en_ctrl *ct = c->ctrl;
            uint64_t s = __atomic_load_n(&ct->req_seq, __ATOMIC_ACQUIRE);
            if (s == c->last_seq) continue;
            uint64_t t0 = now_ns();
            did = 1;
            h->last_activity_ns = t0;
            h->nreq++;
            if (h->cfg.hang_at > 0 && h->nreq == (uint64_t)h->cfg.hang_at) {
                en_log("test hook: reader hangs at request %llu", (unsigned long long)h->nreq);
                while (!__atomic_load_n(&h->stop, __ATOMIC_ACQUIRE)) usleep(10000); /* never answers, but en_stop() can still end it */
                return NULL;
            }
            if (h->cfg.delay_us > 0) usleep((useconds_t)h->cfg.delay_us);
            if (s != c->last_seq + 1)
                en_fatal(h, EN_ERR_PROTO, "doorbell sequence jumped %llu -> %llu (a request was lost)", (unsigned long long)c->last_seq, (unsigned long long)s);
            else if (__atomic_load_n(&ct->status, __ATOMIC_ACQUIRE) == 0)
                serve_request(h, c, s, t0);
            /* publish even after an error so the GPU does not hang; the dequant kernel sees status != 0 and writes NaN */
            sys_barrier();
            __atomic_store_n(&ct->done_seq, s, __ATOMIC_RELEASE);
            c->last_seq = s;
            uint64_t t1 = now_ns(), us = (t1 - t0) / 1000;
            ct->served++;
            ct->last_ns = t1;
            h->st.lat_hist[log2_bucket(us)]++;
            h->st.lat_sum_us += us;
            if (us > h->st.lat_max_us) h->st.lat_max_us = us;
            c->pend_since_ns = 0;
        }
        if (nchan) h->chans[0].ctrl->heartbeat_ns = t;
        if (did) {
            last_active = now_ns();
            continue;
        }
        switch (h->cfg.poll) {
            case EN_POLL_SPIN:
                cpu_relax();
                break;
            case EN_POLL_SLEEP: {
                struct timespec ts = {0, (long)h->cfg.sleep_us * 1000L};
                nanosleep(&ts, NULL);
                break;
            }
            default:
                if (t - last_active < (uint64_t)h->cfg.spin_us * 1000ull) {
                    cpu_relax();
                } else {
                    struct timespec ts = {0, (long)h->cfg.sleep_us * 1000L};
                    nanosleep(&ts, NULL);
                }
        }
    }
    return NULL;
}

static uint64_t xorshift(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static void *keepwarm_main(void *arg) {
    en_handle *h = arg;
    prctl(PR_SET_NAME, "engram-nvme-kw", 0, 0, 0);
    prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, 4096)) return NULL;
    int fd = -1;
    uint64_t span_pages = 0, base = 0, rng = 0x9E3779B97F4A7C15ull;
    uint64_t idle_ns = (uint64_t)h->cfg.keepwarm_idle_ms * 1000000ull;
    while (!__atomic_load_n(&h->stop, __ATOMIC_ACQUIRE)) {
        uint64_t t = now_ns();
        if (fd < 0 && __atomic_load_n(&h->nchan, __ATOMIC_ACQUIRE) > 0) {
            en_chan *c0 = &h->chans[0];
            fd = open(c0->path, O_RDONLY | O_DIRECT | O_CLOEXEC);
            base = (c0->values_off + c0->row_lo * EN_ROW_V + 4095) & ~4095ull;
            uint64_t end = (c0->values_off + c0->row_hi * EN_ROW_V) & ~4095ull;
            span_pages = end > base ? (end - base) / 4096 : 0;
        }
        int active = fd >= 0 && span_pages > 1 && (t - __atomic_load_n(&h->last_activity_ns, __ATOMIC_RELAXED)) < idle_ns;
        if (!active) {
            struct timespec ts = {0, 5 * 1000 * 1000};
            nanosleep(&ts, NULL);
            continue;
        }
        uint64_t off = base + (xorshift(&rng) % span_pages) * 4096;
        if (pread(fd, buf, 4096, (off_t)off) > 0) __atomic_add_fetch(&h->st.keepwarm_reads, 1, __ATOMIC_RELAXED);
        struct timespec ts = {h->cfg.keepwarm_us / 1000000, (long)(h->cfg.keepwarm_us % 1000000) * 1000L};
        nanosleep(&ts, NULL);
    }
    if (fd >= 0) close(fd);
    free(buf);
    return NULL;
}

static void collect_stall(en_handle *h) {
    pthread_mutex_lock(&h->st_mu);
    int nchan = __atomic_load_n(&h->nchan, __ATOMIC_ACQUIRE);
    for (int i = 0; i < nchan; i++) {
        en_chan *c = &h->chans[i];
        uint64_t top = c->stall_seen;
        for (int k = 0; k < EN_STALL_N; k++) {
            uint64_t e = __atomic_load_n(&c->ctrl->stall[k], __ATOMIC_RELAXED);
            uint64_t seq = e >> 32, spins = e & 0xffffffffull;
            if (seq > c->stall_seen && seq <= c->last_seq) {
                h->st.stall_hist[log2_bucket(spins)]++;
                h->st.stall_n++;
                if (seq > top) top = seq;
            }
        }
        c->stall_seen = top;
    }
    pthread_mutex_unlock(&h->st_mu);
}

static void *watchdog_main(void *arg) {
    en_handle *h = arg;
    prctl(PR_SET_NAME, "engram-nvme-wd", 0, 0, 0);
    uint64_t timeout_ns = (uint64_t)h->cfg.timeout_ms * 1000000ull;
    uint64_t last_collect = now_ns(), last_ident = now_ns();
    while (!__atomic_load_n(&h->stop, __ATOMIC_ACQUIRE)) {
        struct timespec ts = {0, 20 * 1000 * 1000};
        nanosleep(&ts, NULL);
        uint64_t t = now_ns();
        int nchan = __atomic_load_n(&h->nchan, __ATOMIC_ACQUIRE);
        for (int i = 0; i < nchan; i++) {
            en_chan *c = &h->chans[i];
            en_ctrl *ct = c->ctrl;
            uint32_t st = __atomic_load_n(&ct->status, __ATOMIC_ACQUIRE);
            if (st != 0) {
                en_fatal(h, (int)st, "channel %d (%s) reports status %u", i, c->path, st);
                continue;
            }
            if (__atomic_load_n(&ct->gpu_abort, __ATOMIC_ACQUIRE)) {
                en_fatal(h, EN_ERR_GPU_ABORT, "channel %d: the GPU wait kernel gave up (timeout or a wait without a new request)", i);
                continue;
            }
            uint64_t rq = __atomic_load_n(&ct->req_seq, __ATOMIC_ACQUIRE), dn = __atomic_load_n(&ct->done_seq, __ATOMIC_ACQUIRE);
            if (rq != dn) {
                if (c->pend_seq != rq) {
                    c->pend_seq = rq;
                    c->pend_since_ns = t;
                } else if (t - c->pend_since_ns > timeout_ns) {
                    en_fatal(h, EN_ERR_TIMEOUT, "channel %d: request %llu pending for %llu ms (reader hung or storage stalled)", i,
                             (unsigned long long)rq, (unsigned long long)((t - c->pend_since_ns) / 1000000ull));
                    c->pend_since_ns = t;
                }
            } else {
                c->pend_seq = 0;
            }
        }
        if (t - last_collect > 1000000000ull) {
            collect_stall(h);
            last_collect = t;
        }
        if (t - last_ident > 5000000000ull) { /* the shard must still be the file we opened */
            last_ident = t;
            for (int i = 0; i < nchan; i++) {
                struct stat sb;
                en_chan *c = &h->chans[i];
                if (fstat(c->fd, &sb) != 0) continue;
                int64_t mt = (int64_t)sb.st_mtim.tv_sec * 1000000000ll + sb.st_mtim.tv_nsec;
                if ((uint64_t)sb.st_ino != c->st_ino || (uint64_t)sb.st_size != c->st_size || mt != c->st_mtime_ns)
                    en_fatal(h, EN_ERR_FILE_CHANGED, "channel %d: %s changed under the engine", i, c->path);
            }
        }
    }
    return NULL;
}

/* ---- public API ------------------------------------------------------------------------------------------------- */
int en_abi_version(void) { return EN_ABI_VERSION; }

void en_ctrl_layout(uint32_t *out) { /* offsets for the Python binding to cross-check */
    out[0] = (uint32_t)offsetof(en_ctrl, req_seq);
    out[1] = (uint32_t)offsetof(en_ctrl, req_ntok);
    out[2] = (uint32_t)offsetof(en_ctrl, done_seq);
    out[3] = (uint32_t)offsetof(en_ctrl, status);
    out[4] = (uint32_t)offsetof(en_ctrl, gpu_abort);
    out[5] = (uint32_t)offsetof(en_ctrl, heartbeat_ns);
    out[6] = (uint32_t)offsetof(en_ctrl, stall);
    out[7] = (uint32_t)sizeof(en_ctrl);
    out[8] = EN_STALL_N;
    out[9] = (uint32_t)sizeof(en_cfg);
    out[10] = (uint32_t)sizeof(en_chan_spec);
    out[11] = (uint32_t)sizeof(en_stats);
}

/* Is Linux native AIO usable here (not blocked by seccomp)? 1 = yes. */
int en_probe_aio(void) {
    aio_context_t ctx = 0;
    if (sys_io_setup(8, &ctx) != 0) return 0;
    sys_io_destroy(ctx);
    return 1;
}

en_handle *en_create(const en_cfg *cfg, char *err, size_t errlen) {
    if (!cfg || cfg->abi != EN_ABI_VERSION) {
        set_err(err, errlen, "ABI mismatch (library %d, caller %d)", EN_ABI_VERSION, cfg ? cfg->abi : -1);
        return NULL;
    }
    en_handle *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    h->cfg = *cfg;
    if (h->cfg.queue_depth < 1) h->cfg.queue_depth = 1;
    if (h->cfg.queue_depth > 4096) h->cfg.queue_depth = 4096;
    if (h->cfg.nthreads < 1) h->cfg.nthreads = 1;
    if (h->cfg.nthreads > 64) h->cfg.nthreads = 64;
    if (h->cfg.timeout_ms < 100) h->cfg.timeout_ms = 100;
    if (h->cfg.sleep_us < 1) h->cfg.sleep_us = 20;
    if (h->cfg.spin_us < 0) h->cfg.spin_us = 0;
    if (h->cfg.keepwarm_idle_ms < 1) h->cfg.keepwarm_idle_ms = 5000;
    pthread_mutex_init(&h->st_mu, NULL);
    if (h->cfg.max_rows < 1) {
        set_err(err, errlen, "cfg.max_rows must be set (rows per slot of the largest channel)");
        free(h);
        return NULL;
    }
    h->jobs_cap = 2u * (uint32_t)h->cfg.max_rows;
    h->dup_cap = (uint32_t)h->cfg.max_rows;
    h->ht_cap = 1;
    while (h->ht_cap < 2u * (uint32_t)h->cfg.max_rows) h->ht_cap <<= 1;
    h->jobs = calloc(h->jobs_cap, sizeof(en_job));
    h->dup_dst = calloc(h->dup_cap, sizeof(uint32_t));
    h->dup_src = calloc(h->dup_cap, sizeof(uint32_t));
    h->ht_key = calloc(h->ht_cap, sizeof(uint32_t));
    h->ht_val = calloc(h->ht_cap, sizeof(uint32_t));
    if (!h->jobs || !h->dup_dst || !h->dup_src || !h->ht_key || !h->ht_val) {
        set_err(err, errlen, "out of memory (request scratch)");
        return NULL;
    }
    h->qd = (uint32_t)h->cfg.queue_depth;
    h->align = 512;
    int want = h->cfg.backend;
    h->backend = EN_BACKEND_THREADS;
    if (want == EN_BACKEND_AUTO || want == EN_BACKEND_AIO) {
        if (sys_io_setup(h->qd, &h->ctx) == 0) {
            h->aio_ready = 1;
            h->backend = EN_BACKEND_AIO;
        } else if (want == EN_BACKEND_AIO) {
            set_err(err, errlen, "io_setup failed: %s (AIO blocked or fs.aio-max-nr exhausted)", strerror(errno));
            free(h);
            return NULL;
        }
    }
    if (h->backend == EN_BACKEND_AIO) {
        h->iocbs = calloc(h->qd, sizeof(struct iocb));
        h->iocb_ptrs = calloc(EN_SUBMIT_MAX, sizeof(struct iocb *));
        h->events = calloc(EN_EVENTS_MAX, sizeof(struct io_event));
        h->free_slots = calloc(h->qd, sizeof(uint32_t));
        if (posix_memalign((void **)&h->win, 4096, (size_t)h->qd * EN_WINSLOT)) h->win = NULL;
        if (!h->iocbs || !h->iocb_ptrs || !h->events || !h->free_slots || !h->win) {
            set_err(err, errlen, "out of memory (AIO state)");
            return NULL;
        }
        for (uint32_t i = 0; i < h->qd; i++) h->free_slots[i] = i;
        h->nfree = h->qd;
    } else {
        en_pool *p = &h->pool;
        p->h = h;
        p->nworkers = h->cfg.nthreads;
        pthread_mutex_init(&p->mu, NULL);
        pthread_cond_init(&p->cv, NULL);
        if (posix_memalign((void **)&p->bufs, 4096, (size_t)(p->nworkers + 1) * EN_WINSLOT)) p->bufs = NULL;
        p->th = calloc((size_t)p->nworkers, sizeof(pthread_t));
        if (!p->bufs || !p->th) {
            set_err(err, errlen, "out of memory (thread pool)");
            return NULL;
        }
        for (int i = 0; i < p->nworkers; i++) {
            void **a = malloc(2 * sizeof(void *));
            a[0] = p;
            a[1] = (void *)(intptr_t)i;
            if (pthread_create(&p->th[i], NULL, pool_worker, a) != 0) {
                set_err(err, errlen, "pthread_create failed");
                return NULL;
            }
        }
    }
    h->st.backend = h->backend;
    return h;
}

int en_backend(en_handle *h) { return h ? h->backend : 0; }

static int probe_align(int fd, uint32_t *align_out) {
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, 8192)) return -ENOMEM;
    uint32_t a = 512;
#if defined(STATX_DIOALIGN)
    struct statx sx;
    if (statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 && (sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align) {
        a = sx.stx_dio_offset_align;
        if (sx.stx_dio_mem_align > 4096) { /* we only guarantee 4096-aligned memory */
            free(buf);
            return -EINVAL;
        }
    }
#endif
    int rc = -EINVAL;
    for (; a <= 4096; a <<= 1) { /* a real O_DIRECT read at that alignment decides */
        ssize_t r = pread(fd, buf, a, 0);
        if (r > 0) {
            *align_out = a;
            rc = 0;
            break;
        }
    }
    free(buf);
    return rc;
}

int en_add_channel(en_handle *h, const en_chan_spec *s, char *err, size_t errlen) {
    int n = h->nchan;
    if (n >= EN_MAX_CHAN) {
        set_err(err, errlen, "too many channels");
        return -1;
    }
    if (!s->ctrl || !s->req_idx || !s->stage_v || !s->stage_s || !s->cap_rows || !s->local_heads || !s->path) {
        set_err(err, errlen, "incomplete channel spec");
        return -1;
    }
    if (((uintptr_t)s->ctrl & 63) || s->row_hi <= s->row_lo || s->row_hi > s->num_rows || s->row_hi > 0x7fffffffull) {
        set_err(err, errlen, "bad channel geometry (rows [%llu,%llu) of %llu)", (unsigned long long)s->row_lo, (unsigned long long)s->row_hi, (unsigned long long)s->num_rows);
        return -1;
    }
    en_chan *c = &h->chans[n];
    memset(c, 0, sizeof *c);
    snprintf(c->path, sizeof c->path, "%s", s->path);
    c->fd = open(s->path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (c->fd < 0) {
        set_err(err, errlen, "open(%s, O_DIRECT): %s", s->path, strerror(errno));
        return -1;
    }
    c->fd_buf = -1;
    if (h->cfg.verify_pct > 0) c->fd_buf = open(s->path, O_RDONLY | O_CLOEXEC);
    struct stat sb;
    if (fstat(c->fd, &sb) != 0) {
        set_err(err, errlen, "fstat: %s", strerror(errno));
        return -1;
    }
    uint64_t vend = s->values_off + s->num_rows * EN_ROW_V, send = s->scales_off + s->num_rows * EN_ROW_S;
    if ((uint64_t)sb.st_size < vend || (uint64_t)sb.st_size < send) {
        set_err(err, errlen, "%s is smaller (%lld B) than the tables need (%llu / %llu B)", s->path, (long long)sb.st_size, (unsigned long long)vend, (unsigned long long)send);
        return -1;
    }
    uint32_t a = 512;
    int pr = probe_align(c->fd, &a);
    if (pr != 0) {
        set_err(err, errlen, "O_DIRECT read probe failed on %s: %s", s->path, strerror(-pr));
        return -1;
    }
    if (a > h->align) h->align = a;
    c->st_ino = sb.st_ino;
    c->st_size = (uint64_t)sb.st_size;
    c->st_mtime_ns = (int64_t)sb.st_mtim.tv_sec * 1000000000ll + sb.st_mtim.tv_nsec;
    c->values_off = s->values_off;
    c->scales_off = s->scales_off;
    c->num_rows = s->num_rows;
    c->row_lo = s->row_lo;
    c->row_hi = s->row_hi;
    c->ctrl = s->ctrl;
    c->req_idx = s->req_idx;
    c->stage_v = s->stage_v;
    c->stage_s = s->stage_s;
    c->cap_rows = s->cap_rows;
    c->local_heads = s->local_heads;
    c->last_seq = __atomic_load_n(&s->ctrl->req_seq, __ATOMIC_ACQUIRE);
    c->stall_seen = c->last_seq;
    if (s->cap_rows > (uint32_t)h->cfg.max_rows) {
        set_err(err, errlen, "channel capacity %u rows exceeds cfg.max_rows %d", s->cap_rows, h->cfg.max_rows);
        return -1;
    }
    h->st.align = h->align;
    __atomic_store_n(&h->nchan, n + 1, __ATOMIC_RELEASE);
    h->st.nchan = n + 1;
    return n;
}

int en_start(en_handle *h, char *err, size_t errlen) {
    if (h->started) return 0;
    if (pthread_create(&h->reader, NULL, reader_main, h) != 0 || pthread_create(&h->watchdog, NULL, watchdog_main, h) != 0) {
        set_err(err, errlen, "pthread_create failed");
        return -1;
    }
    if (h->cfg.keepwarm_us > 0 && pthread_create(&h->keepwarm, NULL, keepwarm_main, h) != 0) {
        set_err(err, errlen, "pthread_create (keep-warm) failed");
        return -1;
    }
    h->started = 1;
    return 0;
}

void en_stop(en_handle *h) {
    if (!h) return;
    __atomic_store_n(&h->stop, 1, __ATOMIC_RELEASE);
    if (h->started) {
        pthread_join(h->reader, NULL);
        pthread_join(h->watchdog, NULL);
        if (h->cfg.keepwarm_us > 0) pthread_join(h->keepwarm, NULL);
        h->started = 0;
    }
    if (h->backend == EN_BACKEND_THREADS) {
        en_pool *p = &h->pool;
        pthread_mutex_lock(&p->mu);
        p->stop = 1;
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        for (int i = 0; i < p->nworkers; i++) pthread_join(p->th[i], NULL);
        p->nworkers = 0;
    } else if (h->aio_ready) {
        sys_io_destroy(h->ctx);
        h->aio_ready = 0;
    }
}

void en_destroy(en_handle *h) {
    if (!h) return;
    en_stop(h);
    int nchan = h->nchan;
    for (int i = 0; i < nchan; i++) {
        if (h->chans[i].fd >= 0) close(h->chans[i].fd);
        if (h->chans[i].fd_buf >= 0) close(h->chans[i].fd_buf);
    }
    free(h->jobs);
    free(h->ht_key);
    free(h->ht_val);
    free(h->dup_dst);
    free(h->dup_src);
    free(h->iocbs);
    free(h->iocb_ptrs);
    free(h->events);
    free(h->free_slots);
    free(h->win);
    free(h->pool.bufs);
    free(h->pool.th);
    free(h);
}

void en_get_stats(en_handle *h, en_stats *out) {
    collect_stall(h);
    *out = h->st;
    out->backend = h->backend;
    out->nchan = h->nchan;
    out->align = h->align;
}

/* ---- hooks for CPU-only tests: they play the GPU's part (same memory-ordering as the Triton kernels) --------------- */
void en_test_gpu_publish(en_chan_spec *s, const int32_t *ids, uint32_t n_ids, uint32_t ntok, uint64_t seq) {
    int32_t *dst = s->req_idx + (size_t)(seq & 1) * s->cap_rows;
    memcpy(dst, ids, (size_t)n_ids * sizeof(int32_t));
    s->ctrl->req_ntok = ntok;
    __atomic_store_n(&s->ctrl->req_seq, seq, __ATOMIC_RELEASE);
}

/* wait (acquire) until done_seq >= seq or status != 0; returns 0 ok, 1 status set, 2 timeout */
int en_test_wait_done(en_chan_spec *s, uint64_t seq, int timeout_ms) {
    uint64_t t0 = now_ns(), lim = (uint64_t)timeout_ms * 1000000ull;
    for (;;) {
        if (__atomic_load_n(&s->ctrl->done_seq, __ATOMIC_ACQUIRE) >= seq) return __atomic_load_n(&s->ctrl->status, __ATOMIC_ACQUIRE) ? 1 : 0;
        if (__atomic_load_n(&s->ctrl->status, __ATOMIC_ACQUIRE)) return 1;
        if (now_ns() - t0 > lim) return 2;
        cpu_relax();
    }
}
