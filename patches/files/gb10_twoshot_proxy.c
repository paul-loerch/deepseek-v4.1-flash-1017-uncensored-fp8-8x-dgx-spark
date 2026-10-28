// SPDX-License-Identifier: Apache-2.0
//
// RDMA proxy + CPU reducer for the GB10 two-shot RoCE all-reduce (gb10_twoshot.py).
//
// Derives from FlashInfer's b12x RoCEnante proxy (_roce_proxy.c, Apache-2.0): the verbs setup, queue-pair connect,
// work-request posting, completion handling and the doorbell loop with its catch-up logic are adapted from it.  The
// two-phase (reduce-scatter + all-gather) protocol, the CPU reduction and the extra error channel are new.
//
// One rank owns one pinned host region (see gb10ts_layout):
//
//   send[slot]               staged by the local GPU kernel (the full message)
//   recv1[src][slot]         phase 1: src's copy of MY chunk, RDMA-written by src
//   recv2[src][slot]         phase 2: src's REDUCED chunk (recv2[me][slot] is written by this thread)
//   flag1[src][slot][rail]   u64 progress word written by src after each phase-1 piece for me: (seq << 8) | pieces_done
//   flag2[src][slot][rail]   u32 sequence number written by src after its reduced chunk (flag2[me] by this thread)
//   ctrl                     doorbell {seq, nbytes, per-slot nbytes, per-slot dtype code} and the error words
//
// For doorbell sequence s (slot = s & 1), P = nbytes / 16 packs and c = ceil(P / W), rank q owns packs
// [q * c, min((q + 1) * c, P)) (possibly empty).  A chunk of n packs is cut into K pieces (K from n and piece_bytes
// alone, so sender and receiver agree; K = 1 when piece_bytes = 0).  This thread
//   1. posts, for every peer p and piece j, an RDMA WRITE of piece j of p's chunk of send[slot] into p's recv1[me][slot]
//      and then an 8-byte inline RDMA WRITE of ((s << 8) | (j + 1)) into p's flag1[me][slot][0] on the same RC queue
//      pair (data unsignaled; the flag of the last piece signaled);
//   2. for each piece j of its own chunk: waits until flag1[p][slot][0] shows sequence s and progress >= j + 1 for every
//      peer p (acquire; wall-clock timeout);
//   3. reduces that piece: acc = (float)x_0; acc += (float)x_q for q = 1 .. W-1 in rank order; one round-to-nearest-even
//      to the element type; the result goes to recv2[me][slot] (bit-identical to RoCEnante's GPU reduction);
//   4. posts the reduced piece to every peer's recv2[me][slot]; the 4-byte inline write of s into the peer's
//      flag2[me][slot][0] follows the LAST piece on the same queue pair (in-order delivery: it cannot overtake a piece);
//   5. after the last piece, release-stores s into its own flag2[me][slot][0] (the local GPU waits for flag2 of ALL
//      ranks, itself included).  With K = 1 this is exactly the single-piece protocol of the design brief (the progress
//      word then only ever shows piece 1).
//
// Why two slots cannot be overwritten early: see tests/roce2/NOTES.md ("Slot reuse").
//
// This file is compiled by gb10_twoshot.py at first use with the host gcc and libibverbs; it is plain C (NEON
// intrinsics when available) with no CUDA dependency.  Build flags: -O2 -ffp-contract=off, no fast-math.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define TS_NEON 1
#else
#define TS_NEON 0
#endif

#define TS_MAX_PEERS 16
#define TS_SLOTS 2
#define TS_RAILS 1
#define TS_FLAG_STRIDE 128
#define TS_CTRL_BYTES 256  // line 0: doorbell + GPU-written words; line 1: proxy-written words
#define TS_PORT 1
#define TS_SEND_DEPTH 256
#define TS_SIGNALED_LIMIT 4   // signaled WRs in flight per peer; each follows at most TS_MAX_PIECES + 1 unsignaled WQEs
#define TS_MAX_PIECES 16      // pieces per chunk and op: <= 3 * 16 + 3 WQEs per peer and op, 3 ops of history stay below TS_SEND_DEPTH
#define TS_ABI_VERSION 1
#define TS_OPT_PHASE1_UNSIGNALED 1u  // experiment: the phase-1 flag write is not signaled (phase 2's flag frees the queue)
#define TS_PROF_ROWS 8192
#define TS_PROF_COLS 8

// ctrl words (u32 index).  Line 0 is written by the GPU kernel, line 1 by this thread.
#define CW_BELL 0         // doorbell: newest sequence
#define CW_NBYTES 1       // nbytes of the newest op
#define CW_GPU_ERR_SEQ 2  // GPU wait timed out / aborted at this sequence (0 = none)
#define CW_GPU_ERR_PEER 3
#define CW_NB_SLOT 4      // per-slot nbytes: words 4, 5
#define CW_GPU_ERR_RAIL 6
#define CW_GPU_ERR_CODE 7  // 1 = flag timeout, 2 = proxy failed
#define CW_CODE_SLOT 8     // per-slot dtype code: words 8, 9
#define CW_PROXY_ERR 32    // proxy error code (0 = ok)
#define CW_PROXY_PEER 33
#define CW_PROXY_SEQ 34
#define CW_PROXY_DONE 35  // last sequence fully processed by the proxy

// dtype codes (kernel writes them with the doorbell)
#define TS_DT_BF16 1
#define TS_DT_FP16 2
#define TS_DT_FP32 3

// proxy error codes
#define TS_E_QP 1        // completion error / post failure
#define TS_E_TIMEOUT 2   // peer flag never arrived
#define TS_E_DOORBELL 3  // doorbell skipped more ops than there are slots
#define TS_E_BADOP 4     // size / dtype the proxy cannot handle

typedef struct {
  uint64_t region_addr;
  uint32_t rkey;
  uint16_t lid;
  uint8_t gid[16];
  uint32_t mtu;
  uint32_t qp_num[TS_MAX_PEERS];
} ts_blob_t;

typedef struct {
  int world, rank, gid_index, traffic_class;
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_mr* mr;
  struct ibv_cq* cq;
  struct ibv_qp* qp[TS_MAX_PEERS];
  uint32_t outstanding[TS_MAX_PEERS];  // signaled WRs in flight per peer
  union ibv_gid gid;
  uint16_t lid;
  enum ibv_mtu mtu;

  uint8_t* region;
  size_t region_bytes;
  size_t max_bytes, slot_bytes, cslot_bytes;
  size_t send_off, recv1_off, recv2_off, flag1_off, flag2_off, ctrl_off;

  uint64_t peer_addr[TS_MAX_PEERS];
  uint32_t peer_rkey[TS_MAX_PEERS];

  pthread_t thread;
  atomic_int running;
  atomic_int failed;
  int started;
  int cpu;       // requested core (-1 = none)
  int pinned;    // result of sched_setaffinity (1 = pinned)
  int profile;
  uint64_t timeout_ticks;
  uint32_t last_seq;
  uint64_t ops_done, writes_completed, bytes_posted, reduce_packs;
  char err[512];
  uint64_t prof[TS_PROF_ROWS][TS_PROF_COLS];
  uint64_t arrive[TS_PROF_ROWS][TS_MAX_PEERS];  // profile only: when each peer's phase-1 flag was first seen
  size_t prefetch_bytes;                         // per source and op: lines prefetched when the flag is seen
  size_t piece_bytes;                            // phase 2: reduce + post in pieces of about this size (0 = one piece)
  uint64_t idle_ticks;                           // spin this long after the last doorbell before napping
  uint32_t options;                              // TS_OPT_*
} ts_ctx_t;

void gb10ts_destroy(ts_ctx_t* c);

// ------------------------------------------------------------------------------------------------ time and memory

#if defined(__aarch64__)
static inline uint64_t ts_now(void) {
  uint64_t v;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
  return v;
}
static inline uint64_t ts_freq(void) {
  uint64_t v;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
  return v;
}
// Profiling timestamp: the ISB keeps the counter read from executing before the preceding instructions complete
// (a bare MRS of the counter is not ordered with them and reads early).
static inline uint64_t ts_prof(void) {
  uint64_t v;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v)::"memory");
  return v;
}
#else
static inline uint64_t ts_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static inline uint64_t ts_freq(void) { return 1000000000ull; }
static inline uint64_t ts_prof(void) { return ts_now(); }
#endif

static inline uint32_t ld_relaxed(const void* p) { return __atomic_load_n((const volatile uint32_t*)p, __ATOMIC_RELAXED); }
static inline uint32_t ld_acquire(const void* p) { return __atomic_load_n((const volatile uint32_t*)p, __ATOMIC_ACQUIRE); }
static inline void st_release(void* p, uint32_t v) { __atomic_store_n((volatile uint32_t*)p, v, __ATOMIC_RELEASE); }
static inline void st_relaxed(void* p, uint32_t v) { __atomic_store_n((volatile uint32_t*)p, v, __ATOMIC_RELAXED); }

static void set_err(ts_ctx_t* c, const char* what, int e) {
  snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int gb10ts_abi_version(void) { return TS_ABI_VERSION; }
uint64_t gb10ts_tick_hz(void) { return ts_freq(); }

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// ------------------------------------------------------------------------------------------------ layout

// out = {send_off, recv1_off, recv2_off, flag1_off, flag2_off, ctrl_off, total_bytes, slot_bytes, cslot_bytes,
//        flag_stride, slots, rails, ctrl_bytes}
int gb10ts_layout(int world, uint64_t max_bytes, uint64_t* out) {
  if (world < 2 || world > TS_MAX_PEERS || max_bytes < 16 || (max_bytes % 16) != 0 || max_bytes > ((uint64_t)1 << 31)) {
    return -1;
  }
  uint64_t max_packs = max_bytes / 16;
  uint64_t slot_bytes = align_up(max_bytes, 4096);
  uint64_t cslot_bytes = align_up(((max_packs + (uint64_t)world - 1) / (uint64_t)world) * 16, 256);
  uint64_t send_off = 0;
  uint64_t recv1_off = align_up(send_off + (uint64_t)TS_SLOTS * slot_bytes, 4096);
  uint64_t recv2_off = align_up(recv1_off + (uint64_t)world * TS_SLOTS * cslot_bytes, 4096);
  uint64_t flag1_off = align_up(recv2_off + (uint64_t)world * TS_SLOTS * cslot_bytes, 4096);
  uint64_t flag_bytes = (uint64_t)world * TS_SLOTS * TS_RAILS * TS_FLAG_STRIDE;
  uint64_t flag2_off = align_up(flag1_off + flag_bytes, 4096);
  uint64_t ctrl_off = align_up(flag2_off + flag_bytes, 4096);
  uint64_t total = ctrl_off + TS_CTRL_BYTES;
  out[0] = send_off;
  out[1] = recv1_off;
  out[2] = recv2_off;
  out[3] = flag1_off;
  out[4] = flag2_off;
  out[5] = ctrl_off;
  out[6] = total;
  out[7] = slot_bytes;
  out[8] = cslot_bytes;
  out[9] = TS_FLAG_STRIDE;
  out[10] = TS_SLOTS;
  out[11] = TS_RAILS;
  out[12] = TS_CTRL_BYTES;
  return 0;
}

uint64_t gb10ts_blob_bytes(void) { return sizeof(ts_blob_t); }

// ------------------------------------------------------------------------------------------------ setup

static int open_hca(ts_ctx_t* c, const char* name) {
  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (list == NULL) {
    set_err(c, "ibv_get_device_list", errno);
    return -1;
  }
  struct ibv_device* dev = NULL;
  for (int i = 0; i < num; i++) {
    if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
      dev = list[i];
      break;
    }
  }
  if (dev == NULL) {
    ibv_free_device_list(list);
    snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
    return -1;
  }
  c->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (c->ctx == NULL) {
    set_err(c, "ibv_open_device", errno);
    return -1;
  }
  struct ibv_port_attr port;
  if (ibv_query_port(c->ctx, TS_PORT, &port) != 0) {
    set_err(c, "ibv_query_port", errno);
    return -1;
  }
  if (port.state != IBV_PORT_ACTIVE) {
    snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, TS_PORT);
    return -1;
  }
  c->lid = port.lid;
  c->mtu = port.active_mtu;
  if (ibv_query_gid(c->ctx, TS_PORT, c->gid_index, &c->gid) != 0) {
    set_err(c, "ibv_query_gid", errno);
    return -1;
  }
  c->pd = ibv_alloc_pd(c->ctx);
  if (c->pd == NULL) {
    set_err(c, "ibv_alloc_pd", errno);
    return -1;
  }
  c->mr = ibv_reg_mr(c->pd, c->region, c->region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (c->mr == NULL) {
    set_err(c, "ibv_reg_mr(pinned region)", errno);
    return -1;
  }
  c->cq = ibv_create_cq(c->ctx, TS_SEND_DEPTH * TS_MAX_PEERS, NULL, NULL, 0);
  if (c->cq == NULL) {
    set_err(c, "ibv_create_cq", errno);
    return -1;
  }
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = c->cq;
    attr.recv_cq = c->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = TS_SEND_DEPTH;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 16;
    c->qp[p] = ibv_create_qp(c->pd, &attr);
    if (c->qp[p] == NULL) {
      set_err(c, "ibv_create_qp", errno);
      return -1;
    }
    struct ibv_qp_attr init;
    memset(&init, 0, sizeof(init));
    init.qp_state = IBV_QPS_INIT;
    init.pkey_index = 0;
    init.port_num = TS_PORT;
    init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    int rc = ibv_modify_qp(c->qp[p], &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc != 0) {
      set_err(c, "ibv_modify_qp(INIT)", rc);
      return -1;
    }
  }
  return 0;
}

ts_ctx_t* gb10ts_create(int world, int rank, const char* hca_name, int gid_index, int traffic_class, void* region,
                        uint64_t region_bytes, uint64_t max_bytes, uint64_t timeout_ms, char* err, uint64_t err_len) {
  uint64_t layout[13];
  if (gb10ts_layout(world, max_bytes, layout) != 0 || layout[6] > region_bytes || rank < 0 || rank >= world ||
      traffic_class < 0 || traffic_class > 255 || hca_name == NULL) {
    snprintf(err, err_len, "invalid two-shot runtime geometry");
    return NULL;
  }
  ts_ctx_t* c = calloc(1, sizeof(*c));
  if (c == NULL) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  c->world = world;
  c->rank = rank;
  c->gid_index = gid_index;
  c->traffic_class = traffic_class;
  c->region = region;
  c->region_bytes = region_bytes;
  c->max_bytes = max_bytes;
  c->send_off = layout[0];
  c->recv1_off = layout[1];
  c->recv2_off = layout[2];
  c->flag1_off = layout[3];
  c->flag2_off = layout[4];
  c->ctrl_off = layout[5];
  c->slot_bytes = layout[7];
  c->cslot_bytes = layout[8];
  c->cpu = -1;
  uint64_t hz = ts_freq();
  c->timeout_ticks = (timeout_ms * hz) / 1000u;
  if (open_hca(c, hca_name) != 0) {
    snprintf(err, err_len, "%s", c->err);
    gb10ts_destroy(c);
    return NULL;
  }
  return c;
}

int gb10ts_local_blob(ts_ctx_t* c, void* out, uint64_t out_len) {
  if (out_len < sizeof(ts_blob_t)) {
    return -1;
  }
  ts_blob_t blob;
  memset(&blob, 0, sizeof(blob));
  blob.region_addr = (uint64_t)(uintptr_t)c->region;
  blob.rkey = c->mr->rkey;
  blob.lid = c->lid;
  blob.mtu = (uint32_t)c->mtu;
  memcpy(blob.gid, c->gid.raw, 16);
  for (int p = 0; p < c->world; p++) {
    blob.qp_num[p] = (p == c->rank) ? 0 : c->qp[p]->qp_num;
  }
  memcpy(out, &blob, sizeof(blob));
  return 0;
}

static int connect_qp(ts_ctx_t* c, int p, const ts_blob_t* peer) {
  struct ibv_qp_attr rtr;
  memset(&rtr, 0, sizeof(rtr));
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu = (enum ibv_mtu)(peer->mtu < (uint32_t)c->mtu ? peer->mtu : (uint32_t)c->mtu);
  rtr.dest_qp_num = peer->qp_num[c->rank];
  rtr.rq_psn = 0;
  rtr.max_dest_rd_atomic = 1;
  rtr.min_rnr_timer = 12;
  rtr.ah_attr.is_global = 1;
  rtr.ah_attr.dlid = peer->lid;
  rtr.ah_attr.sl = 0;
  rtr.ah_attr.src_path_bits = 0;
  rtr.ah_attr.port_num = TS_PORT;
  memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid, 16);
  rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
  rtr.ah_attr.grh.hop_limit = 64;
  rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
  rtr.ah_attr.grh.flow_label = 0;
  int rc = ibv_modify_qp(c->qp[p], &rtr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTR)", rc);
    return -1;
  }
  struct ibv_qp_attr rts;
  memset(&rts, 0, sizeof(rts));
  rts.qp_state = IBV_QPS_RTS;
  rts.timeout = 14;
  rts.retry_cnt = 7;
  rts.rnr_retry = 7;
  rts.sq_psn = 0;
  rts.max_rd_atomic = 1;
  rc = ibv_modify_qp(c->qp[p], &rts,
                     IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                         IBV_QP_MAX_QP_RD_ATOMIC);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTS)", rc);
    return -1;
  }
  return 0;
}

int gb10ts_connect(ts_ctx_t* c, const void* blobs, uint64_t blobs_len) {
  if (blobs_len < sizeof(ts_blob_t) * (uint64_t)c->world) {
    snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
    return -1;
  }
  const ts_blob_t* all = (const ts_blob_t*)blobs;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    c->peer_addr[p] = all[p].region_addr;
    c->peer_rkey[p] = all[p].rkey;
    if (connect_qp(c, p, &all[p]) != 0) {
      return -1;
    }
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------ verbs helpers

static volatile uint32_t* ctrl_words(ts_ctx_t* c) { return (volatile uint32_t*)(c->region + c->ctrl_off); }

static void fail(ts_ctx_t* c, uint32_t code, int peer, uint32_t seq) {
  volatile uint32_t* ctrl = ctrl_words(c);
  st_relaxed((void*)&ctrl[CW_PROXY_PEER], (uint32_t)peer);
  st_relaxed((void*)&ctrl[CW_PROXY_SEQ], seq);
  st_release((void*)&ctrl[CW_PROXY_ERR], code);  // the GPU kernel and the host read this word
  atomic_store(&c->failed, 1);
}

static int drain_cq(ts_ctx_t* c) {
  struct ibv_wc wc[32];
  for (;;) {
    int n = ibv_poll_cq(c->cq, 32, wc);
    if (n < 0) {
      set_err(c, "ibv_poll_cq", errno);
      return -1;
    }
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        snprintf(c->err, sizeof(c->err), "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
                 (unsigned)wc[i].wr_id, ibv_wc_status_str(wc[i].status), wc[i].vendor_err, c->last_seq);
        return -1;
      }
      c->outstanding[wc[i].wr_id] -= 1;
      c->writes_completed += 1;
    }
    if (n < 32) {
      return 0;
    }
  }
}

// A flag write (4 or 8 bytes, inline) preceded, when len > 0, by an unsignaled data write, chained so the QP delivers
// the flag after the data.  `signaled` flag writes free the queue entries before them when they complete.
static int post_pair(ts_ctx_t* c, int p, const void* data, uint32_t len, uint64_t remote_data, uint64_t remote_flag,
                     uint64_t value, uint32_t flag_len, int signaled) {
  while (signaled && c->outstanding[p] >= TS_SIGNALED_LIMIT) {
    if (drain_cq(c) != 0) {
      return -1;
    }
    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
      snprintf(c->err, sizeof(c->err), "two-shot proxy stopped with %u writes outstanding to rank %d", c->outstanding[p],
               p);
      return -1;
    }
  }
  uint64_t flag_value = value;  // little endian: the low 4 bytes are the u32 value for a 4-byte flag
  struct ibv_sge flag_sge = {.addr = (uint64_t)(uintptr_t)&flag_value, .length = flag_len, .lkey = 0};
  struct ibv_send_wr flag_wr;
  memset(&flag_wr, 0, sizeof(flag_wr));
  flag_wr.wr_id = (uint64_t)p;
  flag_wr.sg_list = &flag_sge;
  flag_wr.num_sge = 1;
  flag_wr.opcode = IBV_WR_RDMA_WRITE;
  flag_wr.send_flags = (signaled ? IBV_SEND_SIGNALED : 0) | IBV_SEND_INLINE;
  flag_wr.wr.rdma.remote_addr = remote_flag;
  flag_wr.wr.rdma.rkey = c->peer_rkey[p];
  struct ibv_send_wr data_wr;
  struct ibv_sge data_sge;
  struct ibv_send_wr* first = &flag_wr;
  if (len != 0) {
    data_sge = (struct ibv_sge){.addr = (uint64_t)(uintptr_t)data, .length = len, .lkey = c->mr->lkey};
    memset(&data_wr, 0, sizeof(data_wr));
    data_wr.wr_id = (uint64_t)p;
    data_wr.next = &flag_wr;
    data_wr.sg_list = &data_sge;
    data_wr.num_sge = 1;
    data_wr.opcode = IBV_WR_RDMA_WRITE;
    data_wr.wr.rdma.remote_addr = remote_data;
    data_wr.wr.rdma.rkey = c->peer_rkey[p];
    first = &data_wr;
  }
  struct ibv_send_wr* bad = NULL;
  int rc = ibv_post_send(c->qp[p], first, &bad);
  if (rc != 0) {
    set_err(c, "ibv_post_send", rc);
    return -1;
  }
  if (signaled) {
    c->outstanding[p] += 1;
  }
  c->bytes_posted += len;
  return 0;
}

// One unsignaled data write (a phase-2 piece; the flag that completes the chunk follows on the same queue pair).
static int post_data(ts_ctx_t* c, int p, const void* data, uint32_t len, uint64_t remote_data) {
  struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)data, .length = len, .lkey = c->mr->lkey};
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)p;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr.rdma.remote_addr = remote_data;
  wr.wr.rdma.rkey = c->peer_rkey[p];
  struct ibv_send_wr* bad = NULL;
  int rc = ibv_post_send(c->qp[p], &wr, &bad);
  if (rc != 0) {
    set_err(c, "ibv_post_send", rc);
    return -1;
  }
  c->bytes_posted += len;
  return 0;
}

// ------------------------------------------------------------------------------------------------ reduction

#if TS_NEON

// bf16 -> fp32 is a shift: the low/high four elements of a pack of eight.
static inline float32x4_t bf16_lo(uint16x8_t v) { return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)); }
static inline float32x4_t bf16_hi(uint16x8_t v) { return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16)); }

// fp32 -> bf16, round to nearest even; NaN -> 0x7FFF like CUDA's cvt.rn.bf16.f32.  Integer arithmetic on the bit
// pattern, so the result does not depend on the FPCR (flush-to-zero, default-NaN) of this thread.
static inline uint16x4_t f32_to_bf16(float32x4_t a) {
  uint32x4_t u = vreinterpretq_u32_f32(a);
  uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
  uint32x4_t r = vshrq_n_u32(vaddq_u32(vaddq_u32(u, vdupq_n_u32(0x7fffu)), lsb), 16);
  uint32x4_t is_nan = vcgtq_u32(vandq_u32(u, vdupq_n_u32(0x7fffffffu)), vdupq_n_u32(0x7f800000u));
  r = vbslq_u32(is_nan, vdupq_n_u32(0x7fffu), r);
  return vmovn_u32(r);
}

// fp16 <-> fp32 conversions and the NaN canonicalization (0x7FFF, like cvt.rn.f16.f32).
static inline float32x4_t f16_lo(uint16x8_t v) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v))); }
static inline float32x4_t f16_hi(uint16x8_t v) { return vcvt_high_f32_f16(vreinterpretq_f16_u16(v)); }
static inline uint16x4_t f32_to_f16(float32x4_t a) {
  uint16x4_t h = vreinterpret_u16_f16(vcvt_f16_f32(a));
  uint32x4_t is_nan = vmvnq_u32(vceqq_f32(a, a));  // true for NaN lanes
  return vbsl_u16(vmovn_u32(is_nan), vdup_n_u16(0x7fffu), h);
}

#define TS_REDUCE_BODY(WCONST, LOADLO, LOADHI, NARROW)                                  \
  for (size_t i = 0; i < npacks; i++) {                                                 \
    uint16x8_t v = vld1q_u16((const uint16_t*)(src[0] + 16 * i));                       \
    float32x4_t lo = LOADLO(v), hi = LOADHI(v);                                         \
    for (int q = 1; q < (WCONST); q++) {                                                \
      v = vld1q_u16((const uint16_t*)(src[q] + 16 * i));                                \
      lo = vaddq_f32(lo, LOADLO(v));                                                    \
      hi = vaddq_f32(hi, LOADHI(v));                                                    \
    }                                                                                   \
    vst1q_u16((uint16_t*)(dst + 16 * i), vcombine_u16(NARROW(lo), NARROW(hi)));         \
  }

static inline __attribute__((always_inline)) void reduce_bf16_w(const uint8_t* const* src, uint8_t* dst, size_t npacks,
                                                                const int w) {
  TS_REDUCE_BODY(w, bf16_lo, bf16_hi, f32_to_bf16)
}
static inline __attribute__((always_inline)) void reduce_f16_w(const uint8_t* const* src, uint8_t* dst, size_t npacks,
                                                               const int w) {
  TS_REDUCE_BODY(w, f16_lo, f16_hi, f32_to_f16)
}
static inline __attribute__((always_inline)) void reduce_f32_w(const uint8_t* const* src, uint8_t* dst, size_t npacks,
                                                               const int w) {
  for (size_t i = 0; i < npacks; i++) {
    float32x4_t acc = vld1q_f32((const float*)(src[0] + 16 * i));
    for (int q = 1; q < w; q++) {
      acc = vaddq_f32(acc, vld1q_f32((const float*)(src[q] + 16 * i)));
    }
    // every NaN result becomes the canonical 0x7FFFFFFF the GPU produces (the CPU would keep payloads / 0x7FC00000)
    uint32x4_t is_nan = vmvnq_u32(vceqq_f32(acc, acc));
    acc = vbslq_f32(is_nan, vreinterpretq_f32_u32(vdupq_n_u32(0x7fffffffu)), acc);
    vst1q_f32((float*)(dst + 16 * i), acc);
  }
}

#define TS_DISPATCH(FN, src, dst, npacks, w)                                  \
  switch (w) {                                                                \
    case 2: FN(src, dst, npacks, 2); break;                                   \
    case 3: FN(src, dst, npacks, 3); break;                                   \
    case 4: FN(src, dst, npacks, 4); break;                                   \
    case 5: FN(src, dst, npacks, 5); break;                                   \
    case 6: FN(src, dst, npacks, 6); break;                                   \
    case 7: FN(src, dst, npacks, 7); break;                                   \
    case 8: FN(src, dst, npacks, 8); break;                                   \
    default: FN(src, dst, npacks, w); break;                                  \
  }

#else  // scalar fallback (same arithmetic, element by element)

static inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline uint16_t f32_to_bf16_s(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fffu;
  }
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static void reduce_bf16_generic(const uint8_t* const* src, uint8_t* dst, size_t npacks, int w) {
  for (size_t i = 0; i < npacks * 8; i++) {
    float acc = bf16_to_f32(((const uint16_t*)src[0])[i]);
    for (int q = 1; q < w; q++) {
      acc = acc + bf16_to_f32(((const uint16_t*)src[q])[i]);
    }
    ((uint16_t*)dst)[i] = f32_to_bf16_s(acc);
  }
}
static void reduce_f32_generic(const uint8_t* const* src, uint8_t* dst, size_t npacks, int w) {
  for (size_t i = 0; i < npacks * 4; i++) {
    float acc = ((const float*)src[0])[i];
    for (int q = 1; q < w; q++) {
      acc = acc + ((const float*)src[q])[i];
    }
    if (acc != acc) {
      uint32_t canonical = 0x7fffffffu;
      memcpy(&acc, &canonical, 4);
    }
    ((float*)dst)[i] = acc;
  }
}

#endif

// Reduce npacks 16-byte packs from w sources (rank order) into dst.  Returns 0, or -1 for an unsupported dtype.
static int reduce_chunk(uint32_t code, const uint8_t* const* src, uint8_t* dst, size_t npacks, int w) {
#if TS_NEON
  switch (code) {
    case TS_DT_BF16: TS_DISPATCH(reduce_bf16_w, src, dst, npacks, w) return 0;
    case TS_DT_FP16: TS_DISPATCH(reduce_f16_w, src, dst, npacks, w) return 0;
    case TS_DT_FP32: TS_DISPATCH(reduce_f32_w, src, dst, npacks, w) return 0;
    default: return -1;
  }
#else
  switch (code) {
    case TS_DT_BF16: reduce_bf16_generic(src, dst, npacks, w); return 0;
    case TS_DT_FP32: reduce_f32_generic(src, dst, npacks, w); return 0;
    default: return -1;
  }
#endif
}

// Test hook (single process): reduce npacks packs of w sources into dst exactly as the proxy does.
int gb10ts_reduce(uint32_t code, const void* const* src, void* dst, uint64_t npacks, int w) {
  if (w < 2 || w > TS_MAX_PEERS) {
    return -1;
  }
  return reduce_chunk(code, (const uint8_t* const*)src, (uint8_t*)dst, (size_t)npacks, w);
}

// ------------------------------------------------------------------------------------------------ one all-reduce

static inline uint8_t* send_slot(ts_ctx_t* c, uint32_t slot) { return c->region + c->send_off + (size_t)slot * c->slot_bytes; }
static inline size_t slot_index(int src, uint32_t slot) { return (size_t)src * TS_SLOTS + slot; }
static inline uint8_t* recv1_slot(ts_ctx_t* c, int src, uint32_t slot) {
  return c->region + c->recv1_off + slot_index(src, slot) * c->cslot_bytes;
}
static inline uint8_t* recv2_slot(ts_ctx_t* c, int src, uint32_t slot) {
  return c->region + c->recv2_off + slot_index(src, slot) * c->cslot_bytes;
}
static inline size_t flag_off(int src, uint32_t slot) {
  return slot_index(src, slot) * TS_RAILS * TS_FLAG_STRIDE;
}

// ------------------------------------------------------------------------------------------------ one all-reduce
//
// A chunk of n packs is cut into K = pieces_for(n) pieces of at least four packs (one 64-byte line): piece j covers
// packs [piece_start(n, K, j), piece_start(n, K, j + 1)).  Sender and receiver compute the same cut from n alone, so
// piece_bytes is part of the protocol (the runtime checks that all ranks agree).  Phase 1 writes piece j of a peer's
// chunk and then raises the peer's progress word flag1[me][slot] to ((seq << 8) | (j + 1)) on the same queue pair, so
// the receiver may use piece j as soon as it sees a value >= j + 1 with the right sequence in the upper bits.  The
// receiver reduces piece j (all sources) and posts it as phase-2 piece j while pieces j + 1.. are still in flight;
// the phase-2 flag (a plain u32 sequence, the one the GPU kernel waits for) follows the last piece.

static inline uint32_t pieces_for(const ts_ctx_t* c, uint32_t npacks) {
  if (npacks == 0 || c->piece_bytes == 0) {
    return 1;
  }
  uint64_t k = ((uint64_t)npacks * 16u + c->piece_bytes - 1) / c->piece_bytes;
  uint64_t cap = npacks / 4u;
  k = k > TS_MAX_PIECES ? TS_MAX_PIECES : k;
  k = k > cap ? cap : k;
  return k < 1 ? 1u : (uint32_t)k;
}

static inline uint32_t piece_start(uint32_t npacks, uint32_t pieces, uint32_t j) {
  return j >= pieces ? npacks : (uint32_t)(((uint64_t)npacks * j) / pieces) & ~3u;
}

// Prefetch up to `limit` bytes of a source piece into L2: the lines were DMA-written by the NIC or the GPU and are cold.
static inline void prefetch_chunk(const uint8_t* p, size_t bytes, size_t limit) {
  if (bytes > limit) {
    bytes = limit;
  }
  for (size_t off = 0; off < bytes; off += 64) {
    __builtin_prefetch(p + off, 0, 2);
  }
}

static inline uint64_t ld_relaxed64(const void* p) { return __atomic_load_n((const volatile uint64_t*)p, __ATOMIC_RELAXED); }

// Wait until every peer's phase-1 progress word shows piece `need` (1-based) of this op.  Relaxed polls plus one
// acquire fence at the end: the payload loads that follow cannot be satisfied before the flags were observed.  As each
// flag shows up the peer's piece is prefetched (the thread would be spinning anyway); `stamp` (profile mode) receives
// the first-seen time of each peer's piece-1 flag.
static int wait_piece(ts_ctx_t* c, uint32_t slot, uint32_t seq, uint32_t need, const uint8_t* piece_lo, size_t piece_bytes,
                      size_t piece_off, uint64_t* stamp) {
  uint32_t pending = 0;
  for (int p = 0; p < c->world; p++) {
    if (p != c->rank) {
      pending |= 1u << p;
    }
  }
  uint64_t deadline = ts_now() + c->timeout_ticks;
  uint32_t spins = 0;
  while (pending != 0) {
    uint32_t m = pending;
    while (m != 0) {
      int p = __builtin_ctz(m);
      m &= m - 1;
      uint64_t v = ld_relaxed64(c->region + c->flag1_off + flag_off(p, slot));
      if ((uint32_t)(v >> 8) == seq && (v & 0xffu) >= need) {
        pending &= ~(1u << p);
        if (stamp != NULL && need == 1) {
          stamp[p] = ts_prof();
        }
        if (piece_bytes != 0 && c->prefetch_bytes != 0) {
          // a prefetch is only a hint: the loads that matter come after the acquire fence below
          prefetch_chunk(recv1_slot(c, p, slot) + piece_off, piece_bytes, c->prefetch_bytes);
        }
      }
    }
    if (pending != 0 && (++spins & 1023u) == 0) {
      if (drain_cq(c) != 0) {
        fail(c, TS_E_QP, -1, seq);
        return -1;
      }
      if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
        snprintf(c->err, sizeof(c->err), "two-shot proxy stopped while waiting for phase-1 flags (seq %u)", seq);
        fail(c, TS_E_QP, -1, seq);
        return -1;
      }
      if (ts_now() > deadline) {
        int p = __builtin_ctz(pending);
        snprintf(c->err, sizeof(c->err), "timed out waiting for rank %d's phase-1 piece %u (seq %u, slot %u)", p, need, seq,
                 slot);
        fail(c, TS_E_TIMEOUT, p, seq);
        return -1;
      }
    }
  }
  (void)piece_lo;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return 0;
}

static int run_op(ts_ctx_t* c, uint32_t seq, uint32_t nbytes, uint32_t code) {
  const int W = c->world, me = c->rank;
  const uint32_t slot = seq & 1u;
  if (nbytes == 0 || (nbytes % 16) != 0 || nbytes > c->max_bytes) {
    snprintf(c->err, sizeof(c->err), "two-shot payload size must be a positive multiple of 16 bytes within %zu, got %u",
             c->max_bytes, nbytes);
    fail(c, TS_E_BADOP, -1, seq);
    return -1;
  }
  if (code != TS_DT_BF16 && code != TS_DT_FP16 && code != TS_DT_FP32) {
    snprintf(c->err, sizeof(c->err), "unsupported dtype code %u (seq %u)", code, seq);
    fail(c, TS_E_BADOP, -1, seq);
    return -1;
  }
  const uint32_t P = nbytes / 16;
  const uint32_t cp = (P + (uint32_t)W - 1) / (uint32_t)W;
  uint64_t* prof = c->profile ? c->prof[c->ops_done % TS_PROF_ROWS] : NULL;
  uint64_t t0 = prof ? ts_prof() : 0;

  // 1. phase 1: my copy of every peer's chunk, piece by piece (round-robin over the peers), each piece followed by the
  // peer's progress word
  uint8_t* send = send_slot(c, slot);
  uint32_t rounds = 1;
  uint32_t npk[TS_MAX_PEERS], kp[TS_MAX_PEERS];
  for (int p = 0; p < W; p++) {
    uint64_t lo = (uint64_t)p * cp;
    npk[p] = lo >= P ? 0u : (uint32_t)(P - lo < cp ? P - lo : cp);
    kp[p] = pieces_for(c, npk[p]);
    rounds = kp[p] > rounds ? kp[p] : rounds;
  }
  for (uint32_t j = 0; j < rounds; j++) {
    for (int k = 1; k < W; k++) {
      int p = (me + k) % W;
      if (j >= kp[p]) {
        continue;
      }
      uint32_t a = piece_start(npk[p], kp[p], j), b = piece_start(npk[p], kp[p], j + 1);
      uint64_t base = c->peer_addr[p];
      int last = (j + 1 == kp[p]);
      if (post_pair(c, p, send + ((uint64_t)p * cp + a) * 16u, (b - a) * 16u,
                    base + c->recv1_off + slot_index(me, slot) * c->cslot_bytes + (uint64_t)a * 16u,
                    base + c->flag1_off + flag_off(me, slot), ((uint64_t)seq << 8) | (j + 1), 8,
                    last && !(c->options & TS_OPT_PHASE1_UNSIGNALED)) != 0) {
        fail(c, TS_E_QP, p, seq);
        return -1;
      }
    }
  }
  uint64_t t1 = prof ? ts_prof() : 0;

  // my chunk: [my_lo, my_lo + my_packs) packs, K pieces
  const uint64_t my_lo = (uint64_t)me * cp;
  const uint32_t my_packs = npk[me];
  const uint32_t pieces = kp[me];
  uint8_t* mine = recv2_slot(c, me, slot);
  if (my_packs != 0 && c->prefetch_bytes != 0) {
    prefetch_chunk(send + my_lo * 16u, (size_t)my_packs * 16u, c->prefetch_bytes);
  }
  const uint8_t* src[TS_MAX_PEERS];
  for (int q = 0; q < W; q++) {
    src[q] = (q == me) ? send + my_lo * 16u : recv1_slot(c, q, slot);
  }

  // 2.-4. per piece: wait for every peer's piece, reduce in rank order into recv2[me][slot], send it to every peer
  uint64_t t2 = 0, t3 = 0, t4 = 0;
  for (uint32_t j = 0; j < pieces; j++) {
    uint32_t a = piece_start(my_packs, pieces, j), b = piece_start(my_packs, pieces, j + 1);
    if (wait_piece(c, slot, seq, j + 1, NULL, (size_t)(b - a) * 16u, (size_t)a * 16u,
                   prof ? c->arrive[c->ops_done % TS_PROF_ROWS] : NULL) != 0) {
      return -1;
    }
    if (prof && j == 0) {
      t2 = ts_prof();  // every peer's first piece is in
    }
    if (b != a) {
      const uint8_t* srcj[TS_MAX_PEERS];
      for (int q = 0; q < W; q++) {
        srcj[q] = src[q] + (size_t)a * 16u;
      }
      if (reduce_chunk(code, srcj, mine + (size_t)a * 16u, b - a, W) != 0) {
        snprintf(c->err, sizeof(c->err), "unsupported dtype code %u", code);
        fail(c, TS_E_BADOP, -1, seq);
        return -1;
      }
      c->reduce_packs += b - a;
    }
    const int is_last = (j + 1 == pieces);
    if (prof && j == 0) {
      t3 = ts_prof();  // first piece reduced
    }
    for (int k = 1; k < W; k++) {
      int p = (me + k) % W;
      uint64_t base = c->peer_addr[p];
      uint64_t remote = base + c->recv2_off + slot_index(me, slot) * c->cslot_bytes + (uint64_t)a * 16u;
      if (is_last) {
        if (post_pair(c, p, mine + (size_t)a * 16u, (b - a) * 16u, remote, base + c->flag2_off + flag_off(me, slot), seq, 4,
                      1) != 0) {
          fail(c, TS_E_QP, p, seq);
          return -1;
        }
      } else if (post_data(c, p, mine + (size_t)a * 16u, (b - a) * 16u, remote) != 0) {
        fail(c, TS_E_QP, p, seq);
        return -1;
      }
    }
    if (prof && is_last) {
      t4 = ts_prof();  // last piece reduced and posted
    }
  }

  // 5. my own flag2: the local GPU waits for every rank's, mine included.  Release: the reduced data above is
  // visible to the GPU before it sees the flag.
  st_release(c->region + c->flag2_off + flag_off(me, slot), seq);
  st_relaxed((void*)&ctrl_words(c)[CW_PROXY_DONE], seq);
  if (prof) {
    prof[0] = seq;
    prof[1] = t0;
    prof[2] = t1;
    prof[3] = t2;
    prof[4] = t3;
    prof[5] = t4;
    prof[6] = ts_prof();
    prof[7] = nbytes;
  }
  c->ops_done += 1;
  return 0;
}

static void* proxy_main(void* arg) {
  ts_ctx_t* c = (ts_ctx_t*)arg;
#if defined(__aarch64__)
  // Known floating-point environment: round to nearest, no flush-to-zero, no default-NaN mode, no alternate
  // handling.  (The reduction's rounding is integer arithmetic, but the fp32 adds must be plain IEEE.)
  __asm__ volatile("msr fpcr, xzr");
#endif
  if (c->cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c->cpu, &set);
    c->pinned = (sched_setaffinity(0, sizeof(set), &set) == 0) ? 1 : 0;
  }
  volatile uint32_t* ctrl = ctrl_words(c);
  // Keep the proxy hot across the gaps between collectives (sub-millisecond inside a decode step, a few ms between
  // steps): spin on the doorbell for idle_ticks after the last one, then poll with 20 us naps so an idle server does not
  // hold a core (the first collective after a long pause then pays the wake-up, about 70 us).
  uint64_t idle = 0;
  uint64_t idle_since = ts_now();
  int napping = 0;
  const struct timespec nap = {0, 20000};
  while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
    uint32_t seq = ld_acquire((const void*)&ctrl[CW_BELL]);
    if (seq == c->last_seq) {
      idle++;
      if ((idle & 63u) == 0 && drain_cq(c) != 0) {
        fail(c, TS_E_QP, -1, c->last_seq);
        return NULL;
      }
      if ((idle & 4095u) == 0 && !napping && c->idle_ticks != 0 && ts_now() - idle_since > c->idle_ticks) {
        napping = 1;
      }
      if (napping) {
        nanosleep(&nap, NULL);
      }
      continue;
    }
    idle = 0;
    napping = 0;
    // The doorbell is a level.  The GPU kernel of op N only completes after this thread finished op N, so in
    // normal operation exactly one op is pending; a stream-order violation or a descheduled thread can leave up to
    // TS_SLOTS (every send slot is still intact: the kernel of op N+2 cannot start before op N completed).
    uint32_t pending = seq - c->last_seq;
    if (pending > TS_SLOTS) {
      snprintf(c->err, sizeof(c->err), "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
      fail(c, TS_E_DOORBELL, -1, seq);
      return NULL;
    }
    for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
      uint32_t nbytes = ld_relaxed((const void*)&ctrl[CW_NB_SLOT + (s & 1u)]);
      uint32_t code = ld_relaxed((const void*)&ctrl[CW_CODE_SLOT + (s & 1u)]);
      if (run_op(c, s, nbytes, code) != 0) {
        return NULL;
      }
      c->last_seq = s;
    }
    idle_since = ts_now();
  }
  return NULL;
}

int gb10ts_start(ts_ctx_t* c, int cpu, int profile, uint64_t prefetch_bytes, uint64_t piece_bytes, uint64_t idle_ms,
                 uint32_t options) {
  if (atomic_load(&c->running)) {
    return 0;
  }
  if (!c->started) {
    c->last_seq = ld_relaxed((const void*)&ctrl_words(c)[CW_BELL]);
    c->started = 1;
  }
  c->cpu = cpu;
  c->profile = profile;
  c->prefetch_bytes = (size_t)prefetch_bytes;
  c->piece_bytes = (size_t)piece_bytes;
  c->idle_ticks = (idle_ms * ts_freq()) / 1000u;
  c->options = options;
  atomic_store(&c->failed, 0);
  atomic_store(&c->running, 1);
  int rc = pthread_create(&c->thread, NULL, proxy_main, c);
  if (rc != 0) {
    atomic_store(&c->running, 0);
    set_err(c, "pthread_create", rc);
    return -1;
  }
  return 0;
}

void gb10ts_stop(ts_ctx_t* c) {
  if (atomic_exchange(&c->running, 0)) {
    pthread_join(c->thread, NULL);
  }
}

int gb10ts_failed(ts_ctx_t* c) { return atomic_load(&c->failed); }
const char* gb10ts_error(ts_ctx_t* c) { return c->err; }

// 0 ops done, 1 writes completed, 2 last seq, 3 bytes posted, 4 reduced packs, 5 pinned, 6 profile rows available
uint64_t gb10ts_stat(ts_ctx_t* c, int which) {
  switch (which) {
    case 0: return c->ops_done;
    case 1: return c->writes_completed;
    case 2: return c->last_seq;
    case 3: return c->bytes_posted;
    case 4: return c->reduce_packs;
    case 5: return (uint64_t)c->pinned;
    case 6: return c->ops_done < TS_PROF_ROWS ? c->ops_done : TS_PROF_ROWS;
    default: return 0;
  }
}

// Copy the newest profile rows (oldest first) into out (rows * TS_PROF_COLS u64); returns the row count.
uint64_t gb10ts_prof_read(ts_ctx_t* c, uint64_t* out, uint64_t max_rows) {
  uint64_t done = c->ops_done;
  uint64_t n = done < TS_PROF_ROWS ? done : TS_PROF_ROWS;
  if (n > max_rows) {
    n = max_rows;
  }
  for (uint64_t i = 0; i < n; i++) {
    uint64_t idx = (done - n + i) % TS_PROF_ROWS;
    memcpy(out + i * TS_PROF_COLS, c->prof[idx], TS_PROF_COLS * sizeof(uint64_t));
  }
  return n;
}

// Copy the newest per-peer first-seen times of the phase-1 flags (same rows as gb10ts_prof_read; TS_MAX_PEERS u64 each).
uint64_t gb10ts_arrive_read(ts_ctx_t* c, uint64_t* out, uint64_t max_rows) {
  uint64_t done = c->ops_done;
  uint64_t n = done < TS_PROF_ROWS ? done : TS_PROF_ROWS;
  if (n > max_rows) {
    n = max_rows;
  }
  for (uint64_t i = 0; i < n; i++) {
    uint64_t idx = (done - n + i) % TS_PROF_ROWS;
    memcpy(out + i * TS_MAX_PEERS, c->arrive[idx], TS_MAX_PEERS * sizeof(uint64_t));
  }
  return n;
}

void gb10ts_destroy(ts_ctx_t* c) {
  if (c == NULL) {
    return;
  }
  gb10ts_stop(c);
  for (int p = 0; p < TS_MAX_PEERS; p++) {
    if (c->qp[p] != NULL) {
      ibv_destroy_qp(c->qp[p]);
    }
  }
  if (c->cq != NULL) {
    ibv_destroy_cq(c->cq);
  }
  if (c->mr != NULL) {
    ibv_dereg_mr(c->mr);
  }
  if (c->pd != NULL) {
    ibv_dealloc_pd(c->pd);
  }
  if (c->ctx != NULL) {
    ibv_close_device(c->ctx);
  }
  free(c);
}
