# SPDX-License-Identifier: Apache-2.0
"""Test library of the CPU checks (any CPU, python3 >= 3.10 + numpy; NO torch, NO vLLM install, no GPU).

  * a numpy-backed fake `torch` (just the surface the real image files and the patch modules use): bf16 stored as
    uint16 bit patterns (round-to-nearest-even), float8_e4m3fn as uint8 bit patterns, real byte-level .view(dtype)
    reinterpretation, fp32 GEMM accumulation with one rounding per op;
  * an N-thread rank emulation: each thread is one TP rank with its own world; collectives rendezvous by call
    sequence number, every rank must issue the same (op, shape, dtype) at the same index (a mismatch raises
    ScheduleMismatch on all ranks), a missing rank is a HANG (TimeoutError after a few seconds, recorded);
    patch 012's one-shot is emulated as its kernel computes (fp32 sum in rank order 0..W-1, one bf16 rounding).
  * CUDA streams and graphs per rank thread: torch.cuda.Stream / stream() / current_stream(), CUDAGraph +
    torch.cuda.graph(): while a thread captures, nothing communicates (a one-shot is RECORDED, like NCCL p2p in a
    captured stream) and the CUDA calls a real capture forbids raise (item(), .cpu(), synchronize(), Event timing,
    a capture on the default stream, a nested capture, a replay while capturing); replay() issues the recorded
    one-shots as collectives with the eager one-shot's signature and advances the thread's VIRTUAL GPU clock by
    GRAPH_COST_FN(ops, world) ms; torch.cuda.Event reads that clock (eager work costs 0 ms on this fake GPU).
    Per-world hooks: capture_hook(graph) at capture begin, replay_hook(graph) before a replay launches.
  * the caching allocator's capture bookkeeping as torch's CUDAGraph does it (pool registered before a capture begins,
    released only after a successful end; an invalidated capture leaves it registered).
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
import types

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# bit-level helpers
# ---------------------------------------------------------------------------------------------------------------------


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint32)
    nan = np.isnan(a)
    r = np.where(nan, ((u >> 16) | 0x40).astype(np.uint32), r)
    return r.astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _fp8_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


FP8_TABLE = _fp8_table()
FP8_POS = FP8_TABLE[:127].astype(np.float64)        # bytes 0x00..0x7E: 0 .. 448, increasing


def f32_to_fp8_bits(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    x = np.abs(a).astype(np.float64)
    e = np.floor(np.log2(np.where(x > 0, x, 1.0)))
    e = np.maximum(e, -6.0)
    sp = 2.0 ** (e - 3)
    q = np.round(x / sp) * sp                          # half-to-even
    idx = np.searchsorted(FP8_POS, q)
    idx = np.clip(idx, 0, 126).astype(np.uint8)
    out = idx | np.where(np.signbit(a), np.uint8(0x80), np.uint8(0))
    bad = np.isnan(a) | (x >= 464.0)                   # c10: |x| >= 464 -> NaN (no saturation in a plain cast)
    return np.where(bad, np.uint8(0x7F), out).astype(np.uint8)


def fp8_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return FP8_TABLE[np.asarray(b, dtype=np.uint8)]


# ---------------------------------------------------------------------------------------------------------------------
# fake torch
# ---------------------------------------------------------------------------------------------------------------------
class dtype:
    def __init__(self, name, store, itemsize, kind):
        self.name, self.store, self.itemsize, self.kind = name, np.dtype(store), itemsize, kind

    @property
    def is_floating_point(self):
        return self.kind in ("f", "bf16", "fp8")

    def __repr__(self):
        return f"torch.{self.name}"

    def __reduce__(self):
        return (_dtype_by_name, (self.name,))


float32 = dtype("float32", np.float32, 4, "f")
float64 = dtype("float64", np.float64, 8, "f")
float16 = dtype("float16", np.float16, 2, "f")
bfloat16 = dtype("bfloat16", np.uint16, 2, "bf16")
float8_e4m3fn = dtype("float8_e4m3fn", np.uint8, 1, "fp8")
int8 = dtype("int8", np.int8, 1, "i")
uint8 = dtype("uint8", np.uint8, 1, "i")
int16 = dtype("int16", np.int16, 2, "i")
int32 = dtype("int32", np.int32, 4, "i")
int64 = dtype("int64", np.int64, 8, "i")
bool_ = dtype("bool", np.bool_, 1, "b")
_DTYPES = {d.name: d for d in (float32, float64, float16, bfloat16, float8_e4m3fn, int8, uint8, int16, int32, int64,
                               bool_)}


def _dtype_by_name(n):
    return _DTYPES[n]


def _decode(a: np.ndarray, dt: dtype) -> np.ndarray:
    if dt is bfloat16:
        return bf16_bits_to_f32(a)
    if dt is float8_e4m3fn:
        return fp8_bits_to_f32(a)
    if dt is float16:
        return a.astype(np.float32)
    return a


def _encode(v, dt: dtype) -> np.ndarray:
    v = np.asarray(v)
    if dt is bfloat16:
        return f32_to_bf16_bits(v.astype(np.float32))
    if dt is float8_e4m3fn:
        return f32_to_fp8_bits(v.astype(np.float32))
    if dt.kind == "i" and v.dtype.kind == "f":
        v = np.trunc(v)
    return v.astype(dt.store)


_FLOAT_RANK = {"float64": 4, "float32": 3, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1}


def _promote(a: dtype, b: dtype) -> dtype:
    if a is b:
        return a
    if a.kind == "b":
        return b
    if b.kind == "b":
        return a
    af, bf = a.is_floating_point, b.is_floating_point
    if af and bf:
        ra, rb = _FLOAT_RANK[a.name], _FLOAT_RANK[b.name]
        if ra == rb:
            return float32
        return a if ra > rb else b
    if af:
        return a
    if bf:
        return b
    return a if a.itemsize >= b.itemsize else b


class device:
    def __init__(self, spec="cpu", index=None):
        if isinstance(spec, device):
            self.type, self.index = spec.type, spec.index
            return
        s = str(spec)
        if ":" in s:
            t, i = s.split(":")
            self.type, self.index = t, int(i)
        else:
            self.type, self.index = s, (index if index is not None else (0 if s == "cuda" else None))

    def __eq__(self, o):
        o = device(o) if not isinstance(o, device) else o
        return self.type == o.type and (self.index or 0) == (o.index or 0)

    def __hash__(self):
        return hash((self.type, self.index or 0))

    def __repr__(self):
        return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")


_Device = device


class Size(tuple):
    def numel(self):
        return int(np.prod(self)) if len(self) else 1


class _MaxResult(tuple):
    @property
    def values(self):
        return self[0]

    @property
    def indices(self):
        return self[1]


_DEFAULT = {"dtype": float32}
_ptr_counter = [1 << 20]


def _np_index(idx):
    if isinstance(idx, Tensor):
        return idx._a if idx.dtype.kind in ("i", "b") else idx._a
    if isinstance(idx, tuple):
        return tuple(_np_index(i) for i in idx)
    if isinstance(idx, list):
        return [(_np_index(i) if isinstance(i, Tensor) else i) for i in idx]
    return idx


class Tensor:
    __array_priority__ = 1000

    def __init__(self, a: np.ndarray, dt: dtype, dev=None, requires_grad=False):
        self._a = a
        self.dtype = dt
        self.device = device(dev) if dev is not None else device("cpu")
        self.requires_grad = requires_grad

    # ---- basic properties
    @property
    def shape(self):
        return Size(self._a.shape)

    def size(self, d=None):
        return Size(self._a.shape) if d is None else self._a.shape[d]

    def dim(self):
        return self._a.ndim

    @property
    def ndim(self):
        return self._a.ndim

    def numel(self):
        return int(self._a.size)

    nelement = numel

    def flip(self, d):
        return self._new(np.ascontiguousarray(np.flip(self._a, axis=d)))

    def element_size(self):
        return self.dtype.itemsize

    def stride(self, d=None):
        st = tuple(s // self._a.itemsize for s in self._a.strides)
        return st if d is None else st[d]

    def is_contiguous(self):
        return bool(self._a.flags["C_CONTIGUOUS"])

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def data(self):
        return self

    @property
    def T(self):
        return Tensor(self._a.T, self.dtype, self.device)

    def t(self):
        return self.T

    def data_ptr(self):
        return self._a.__array_interface__["data"][0]

    def __len__(self):
        return self._a.shape[0]

    def __repr__(self):
        return f"FakeTensor(shape={tuple(self._a.shape)}, dtype={self.dtype}, device={self.device})"

    # ---- conversion
    def _f(self):
        return _decode(self._a, self.dtype)

    def _new(self, a, dt=None):
        return Tensor(a, dt or self.dtype, self.device)

    def _from_f(self, v, dt=None):
        dt = dt or self.dtype
        return Tensor(_encode(v, dt), dt, self.device)

    def to(self, *args, **kw):
        dt, dev = kw.get("dtype"), kw.get("device")
        for a in args:
            if isinstance(a, dtype):
                dt = a
            elif isinstance(a, (str, device)):
                dev = a
            elif isinstance(a, Tensor):
                dt, dev = a.dtype, a.device
        out = self
        if dt is not None and dt is not self.dtype:
            out = Tensor(_encode(self._f(), dt), dt, self.device)
        if dev is not None and device(dev) != self.device:
            out = Tensor(out._a.copy() if out is self else out._a, out.dtype, dev)
        return out

    def float(self):
        return self.to(float32)

    def double(self):
        return self.to(float64)

    def bfloat16(self):
        return self.to(bfloat16)

    def half(self):
        return self.to(float16)

    def int(self):
        return self.to(int32)

    def long(self):
        return self.to(int64)

    def bool(self):
        return self.to(bool_)

    def cpu(self):
        _not_capturing("Tensor.cpu (device-to-host copy)")
        return self.to(device="cpu")

    def cuda(self):
        return self.to(device="cuda:0")

    def numpy(self):
        _not_capturing("Tensor.numpy")
        if self.dtype in (bfloat16, float8_e4m3fn):
            raise TypeError(f"Got unsupported ScalarType {self.dtype.name}")
        return self._a

    def tolist(self):
        _not_capturing("Tensor.tolist")
        return self._f().tolist()

    def item(self):
        _not_capturing("Tensor.item (synchronising device-to-host copy)")
        if self._a.size != 1:
            raise RuntimeError("a Tensor with more than one element cannot be converted to Scalar")
        v = self._f().reshape(-1)[0]
        return v.item() if hasattr(v, "item") else v

    def __float__(self):
        return float(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def __index__(self):
        return int(self.item())

    # ---- views / copies
    def contiguous(self):
        if self.is_contiguous():
            return self
        return self._new(np.ascontiguousarray(self._a))

    def clone(self):
        return self._new(self._a.copy())

    def detach(self):
        return self

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], dtype):
            nd = shape[0]
            if not self._a.flags["C_CONTIGUOUS"]:
                raise RuntimeError("view(dtype) needs a contiguous tensor in this fake")
            return Tensor(self._a.view(nd.store), nd, self.device)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
            shape = tuple(shape[0])
        if not self._a.flags["C_CONTIGUOUS"]:
            raise RuntimeError("view size is not compatible with input tensor's size and stride")
        return self._new(self._a.reshape(shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)):
            shape = tuple(shape[0])
        return self._new(self._a.reshape(shape))

    def flatten(self):
        return self.reshape(-1)

    def unsqueeze(self, d):
        return self._new(np.expand_dims(self._a, d))

    def squeeze(self, d=None):
        return self._new(np.squeeze(self._a, axis=d) if d is not None else np.squeeze(self._a))

    def transpose(self, a, b):
        return self._new(np.swapaxes(self._a, a, b))

    def __getitem__(self, idx):
        return self._new(self._a[_np_index(idx)])

    def __setitem__(self, idx, value):
        idx = _np_index(idx)
        if isinstance(value, Tensor):
            v = value._a if value.dtype is self.dtype else _encode(value._f(), self.dtype)
        else:
            v = _encode(np.asarray(value, dtype=np.float64 if self.dtype.is_floating_point else None), self.dtype)
        self._a[idx] = v

    def copy_(self, src):
        v = src._a if src.dtype is self.dtype else _encode(src._f(), self.dtype)
        self._a[...] = np.broadcast_to(v, self._a.shape)
        return self

    def fill_(self, val):
        self._a[...] = _encode(np.asarray(val, dtype=np.float64), self.dtype) if self.dtype.is_floating_point \
            else val
        return self

    def zero_(self):
        return self.fill_(0)

    def new_zeros(self, *size, dtype=None, device=None):
        return zeros(*size, dtype=dtype or self.dtype, device=device or self.device)

    def new_empty(self, *size, dtype=None, device=None):
        return zeros(*size, dtype=dtype or self.dtype, device=device or self.device)

    def new_tensor(self, data, dtype=None):
        return tensor(data, dtype=dtype or self.dtype, device=self.device)

    # ---- math
    def _bin(self, o, fn, reverse=False, cmp=False):
        if isinstance(o, Tensor):
            dt = _promote(self.dtype, o.dtype)
            a, b = self._f(), o._f()
        else:
            if self.dtype.kind in ("i", "b") and isinstance(o, float):
                dt = float32
            else:
                dt = self.dtype
            a, b = self._f(), o
        if dt.kind in ("bf16", "fp8") or dt is float16:
            a = np.asarray(a, dtype=np.float32)
            if isinstance(b, np.ndarray):
                b = b.astype(np.float32)
        r = fn(b, a) if reverse else fn(a, b)
        if cmp:
            return Tensor(np.asarray(r, dtype=np.bool_), bool_, self.device)
        if dt.kind in ("f",) and dt is not float16:
            return Tensor(np.asarray(r).astype(dt.store), dt, self.device)
        return Tensor(_encode(np.asarray(r, dtype=np.float32 if dt.is_floating_point else None), dt), dt,
                      self.device)

    def __add__(self, o):
        return self._bin(o, np.add)

    def __radd__(self, o):
        return self._bin(o, np.add, True)

    def __sub__(self, o):
        return self._bin(o, np.subtract)

    def __rsub__(self, o):
        return self._bin(o, np.subtract, True)

    def __mul__(self, o):
        return self._bin(o, np.multiply)

    def __rmul__(self, o):
        return self._bin(o, np.multiply, True)

    def __truediv__(self, o):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._bin(o, np.true_divide)

    def __rtruediv__(self, o):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._bin(o, np.true_divide, True)

    def __pow__(self, o):
        return self._bin(o, np.power)

    def __neg__(self):
        return self._from_f(-self._f()) if self.dtype.is_floating_point else self._new(-self._a)

    def __and__(self, o):
        return self._bin(o, np.logical_and if self.dtype.kind == "b" else np.bitwise_and)

    def __or__(self, o):
        return self._bin(o, np.logical_or if self.dtype.kind == "b" else np.bitwise_or)

    def __invert__(self):
        return self._new(~self._a)

    def __eq__(self, o):
        return self._bin(o, np.equal, cmp=True)

    def __ne__(self, o):
        return self._bin(o, np.not_equal, cmp=True)

    def __lt__(self, o):
        return self._bin(o, np.less, cmp=True)

    def __le__(self, o):
        return self._bin(o, np.less_equal, cmp=True)

    def __gt__(self, o):
        return self._bin(o, np.greater, cmp=True)

    def __ge__(self, o):
        return self._bin(o, np.greater_equal, cmp=True)

    __hash__ = object.__hash__

    def __matmul__(self, o):
        return matmul(self, o)

    def abs(self):
        return self._from_f(np.abs(self._f())) if self.dtype.is_floating_point else self._new(np.abs(self._a))

    def sqrt(self):
        return self._from_f(np.sqrt(self._f()))

    def exp(self):
        return self._from_f(np.exp(self._f()))

    def clamp(self, min=None, max=None):
        v = self._f()
        if min is not None:
            v = np.maximum(v, min)
        if max is not None:
            v = np.minimum(v, max)
        return self._from_f(v) if self.dtype.is_floating_point else self._new(v.astype(self.dtype.store))

    def clamp_min(self, m):
        return self.clamp(min=m)

    def _red(self, fn, dim=None, keepdim=False, out_dt=None):
        v = self._f()
        r = fn(v) if dim is None else fn(v, axis=dim, keepdims=keepdim)
        dt = out_dt or self.dtype
        if dt.is_floating_point:
            return Tensor(_encode(np.asarray(r, dtype=np.float64), dt), dt, self.device)
        return Tensor(np.asarray(r).astype(dt.store), dt, self.device)

    def sum(self, dim=None, keepdim=False):
        out_dt = int64 if self.dtype.kind in ("i", "b") else self.dtype
        return self._red(lambda v, **k: np.sum(v.astype(np.float64) if out_dt.is_floating_point else v, **k),
                         dim, keepdim, out_dt)

    def mean(self, dim=None, keepdim=False):
        return self._red(lambda v, **k: np.mean(v.astype(np.float64), **k), dim, keepdim)

    def amax(self, dim=None, keepdim=False):
        return self._red(np.max, dim, keepdim)

    def amin(self, dim=None, keepdim=False):
        return self._red(np.min, dim, keepdim)

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return self._red(np.max)
        return _MaxResult((self.amax(dim, keepdim), self.argmax(dim, keepdim)))

    def argmax(self, dim=None, keepdim=False):
        v = self._f()
        r = np.argmax(v) if dim is None else np.argmax(v, axis=dim)
        if keepdim and dim is not None:
            r = np.expand_dims(r, dim)
        return Tensor(np.asarray(r, dtype=np.int64), int64, self.device)

    def all(self):
        return Tensor(np.asarray(bool(np.all(self._f()))), bool_, self.device)

    def any(self):
        return Tensor(np.asarray(bool(np.any(self._f()))), bool_, self.device)

    def norm(self):
        return self._red(lambda v: np.sqrt(np.sum(v.astype(np.float64) ** 2)))


class Parameter(Tensor):
    def __new__(cls, data=None, requires_grad=True):
        return Tensor.__new__(cls)

    def __init__(self, data=None, requires_grad=True):
        d = data if data is not None else Tensor(np.zeros((0,), np.float32), float32)
        super().__init__(d._a, d.dtype, d.device, requires_grad)


def _size_args(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, Size)):
        return tuple(size[0])
    return tuple(int(s) for s in size)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    dt = dtype or _DEFAULT["dtype"]
    return Tensor(np.zeros(_size_args(size), dtype=dt.store), dt, device)


empty = zeros


def ones(*size, dtype=None, device=None):
    dt = dtype or _DEFAULT["dtype"]
    return Tensor(_encode(np.ones(_size_args(size)), dt), dt, device)


def full(size, value, dtype=None, device=None):
    dt = dtype or _DEFAULT["dtype"]
    return Tensor(_encode(np.full(_size_args((size,)), value, dtype=np.float64), dt), dt, device)


def zeros_like(t, dtype=None):
    return zeros(*t.shape, dtype=dtype or t.dtype, device=t.device)


empty_like = zeros_like


def ones_like(t, dtype=None):
    return ones(*t.shape, dtype=dtype or t.dtype, device=t.device)


def tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor):
        return data.to(dtype or data.dtype).to(device=device or data.device)
    a = np.asarray(data)
    if dtype is None:
        dtype = float32 if a.dtype.kind == "f" else (int64 if a.dtype.kind in "iu" else bool_)
    return Tensor(_encode(a, dtype), dtype, device)


def arange(*args, dtype=None, device=None):
    a = np.arange(*args)
    dt = dtype or (int64 if a.dtype.kind in "iu" else float32)
    return Tensor(_encode(a, dt), dt, device)


class Generator:
    def __init__(self, device="cpu"):
        self.seed = 0

    def manual_seed(self, s):
        self.seed = int(s)
        return self


def randn(*size, generator=None, dtype=None, device=None):
    rng = np.random.default_rng(generator.seed if generator is not None else 0)
    dt = dtype or _DEFAULT["dtype"]
    return Tensor(_encode(rng.standard_normal(_size_args(size)), dt), dt, device)


def cat(ts, dim=0):
    dt = ts[0].dtype
    for t in ts[1:]:
        dt = _promote(dt, t.dtype)
    a = np.concatenate([(t._a if t.dtype is dt else _encode(t._f(), dt)) for t in ts], axis=dim)
    return Tensor(a, dt, ts[0].device)


def stack(ts, dim=0):
    return Tensor(np.stack([t._a for t in ts], axis=dim), ts[0].dtype, ts[0].device)


def where(cond, a, b):
    av = a._f() if isinstance(a, Tensor) else a
    bv = b._f() if isinstance(b, Tensor) else b
    dt = a.dtype if isinstance(a, Tensor) else (b.dtype if isinstance(b, Tensor) else float32)
    return Tensor(_encode(np.where(cond._a, av, bv), dt), dt, cond.device)


def isfinite(t):
    return Tensor(np.isfinite(t._f()), bool_, t.device)


def equal(a, b):
    return a.shape == b.shape and bool(np.array_equal(a._a, b._a))


def matmul(a, b):
    dt = _promote(a.dtype, b.dtype)
    if dt is float64:
        r = a._f().astype(np.float64) @ b._f().astype(np.float64)
        return Tensor(r, float64, a.device)
    r = (a._f().astype(np.float32) @ b._f().astype(np.float32)).astype(np.float32)
    return Tensor(_encode(r, dt) if dt is not float32 else r, dt, a.device)


mm = matmul


def get_default_dtype():
    return _DEFAULT["dtype"]


def set_default_dtype(d):
    _DEFAULT["dtype"] = d


@contextlib.contextmanager
def _nullctx(*a, **k):
    yield


class _GradCtx:
    def __call__(self, fn=None):
        if callable(fn):
            return fn
        return self

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def no_grad():
    return _GradCtx()


def inference_mode(mode=True):
    return _GradCtx()


# ---- nn
class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, k, v):
        if "_parameters" not in self.__dict__:
            object.__setattr__(self, "_parameters", {})
            object.__setattr__(self, "_modules", {})
        if isinstance(v, Parameter):
            self._modules.pop(k, None)
            self.__dict__.pop(k, None)
            self._parameters[k] = v
        elif isinstance(v, Module):
            self._parameters.pop(k, None)
            self.__dict__.pop(k, None)
            self._modules[k] = v
        else:
            if k in self._parameters and v is None:
                self._parameters[k] = None
                return
            self._parameters.pop(k, None)
            self._modules.pop(k, None)
            object.__setattr__(self, k, v)

    def __getattr__(self, k):
        d = self.__dict__
        if "_parameters" in d and k in d["_parameters"]:
            return d["_parameters"][k]
        if "_modules" in d and k in d["_modules"]:
            return d["_modules"][k]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{k}'")

    def __delattr__(self, k):
        if k in self._parameters:
            del self._parameters[k]
        elif k in self._modules:
            del self._modules[k]
        else:
            object.__delattr__(self, k)

    def __dir__(self):
        return sorted(set(object.__dir__(self)) | set(self._parameters) | set(self._modules))

    def register_parameter(self, name, p):
        self._parameters[name] = p

    def named_parameters(self, prefix=""):
        for k, p in self._parameters.items():
            if p is not None:
                yield (prefix + k, p)
        for mk, m in self._modules.items():
            yield from m.named_parameters(prefix + mk + ".")

    def named_modules(self, prefix=""):
        yield prefix, self
        for mk, m in self._modules.items():
            yield from m.named_modules(prefix + ("." if prefix else "") + mk)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        dt = dtype or _DEFAULT["dtype"]
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(zeros(out_features, in_features, dtype=dt, device=device or "cuda:0"))
        self.bias = Parameter(zeros(out_features, dtype=dt, device=device or "cuda:0")) if bias else None

    def forward(self, x):
        return linear(x, self.weight, self.bias)


class ModuleDict(Module):
    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self._modules[k] = v

    def __getitem__(self, k):
        return self._modules[k]

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def keys(self):
        return self._modules.keys()


def linear(x, w, b=None):
    g = capturing_graph()
    if g is not None:
        g.ops.append(("linear", tuple(w.shape)))
    out = matmul(x, w.T)
    return out if b is None else out + b


def pad(x, p, mode="constant", value=0.0):
    g = capturing_graph()
    if g is not None:
        g.ops.append(("pad", tuple(x.shape)))
    widths = [(0, 0)] * x.ndim
    for i in range(len(p) // 2):
        d = x.ndim - 1 - i
        widths[d] = (int(p[2 * i]), int(p[2 * i + 1]))
    fill = _encode(np.asarray(value, dtype=np.float64), x.dtype) if x.dtype.is_floating_point else value
    a = np.pad(x._a, widths, mode="constant", constant_values=np.asarray(fill).item() if np.ndim(fill) == 0 else fill)
    return Tensor(a, x.dtype, x.device)


# ---- cuda (rev 2: per-thread streams, CUDA-graph capture / replay, a virtual GPU clock)
class _CudaState:
    syncs = 0


_CUDA_TLS = threading.local()        # per rank thread: .stream (current), .capture (graph being captured), .clock


def capturing_graph():
    return getattr(_CUDA_TLS, "capture", None)


def _not_capturing(what: str) -> None:
    if capturing_graph() is not None:
        _CUDA_TLS.invalidated = True          # as CUDA: a prohibited call INVALIDATES the capture sequence
        raise RuntimeError(f"CUDA error: operation not permitted when stream is capturing ({what})")


# (verification of rev 2) the caching allocator's capture bookkeeping, per rank thread (a real process has one per
# device), as torch's CUDAGraph does it: capture_begin registers the capture's private pool (beginAllocateToPool,
# use count + 1) BEFORE cudaStreamBeginCapture; capture_end unregisters it (endAllocateToPool) only AFTER
# cudaStreamEndCapture succeeded; reset() of an ENDED capture drops the use count (releasePool); empty_cache() asserts
# that no capture is registered (captures_underway.empty()). STUCK_ON_INVALIDATED[0] = True models that ordering (an
# invalidated / failed-to-begin capture leaves its pool registered); False models a torch that unregisters first.
STUCK_ON_INVALIDATED = [True]
POOL_EVENTS: list = []                # (rank, event, pool): 'underflow' = a releasePool below zero (a double release)
_POOL_IDS = __import__("itertools").count(1)


def _alloc_state() -> dict:
    st = getattr(_CUDA_TLS, "alloc", None)
    if st is None:
        st = _CUDA_TLS.alloc = {"underway": [], "uc": {}}
    return st


def graph_pool_handle():
    return (0, next(_POOL_IDS))


def _pool_begin(pool) -> None:
    st = _alloc_state()
    st["underway"].append(pool)
    st["uc"][pool] = st["uc"].get(pool, 0) + 1


def _cuda_endAllocateToPool(device, pool) -> None:
    st = _alloc_state()
    if pool not in st["underway"]:
        raise RuntimeError("endAllocatePool: not currently recording to mempool_id")
    st["underway"].remove(pool)


def _cuda_releasePool(device, pool) -> None:
    st = _alloc_state()
    uc = st["uc"].get(pool, 0) - 1
    st["uc"][pool] = uc
    if uc < 0:
        POOL_EVENTS.append((getattr(current_world(), "rank", None), "underflow", pool))
        raise RuntimeError("INTERNAL ASSERT FAILED: uc >= 0 (releasePool)")


def empty_cache() -> None:
    if _alloc_state()["underway"]:
        raise RuntimeError("captures_underway.empty() INTERNAL ASSERT FAILED at c10/cuda/CUDACachingAllocator.cpp "
                           "(a CUDA-graph capture is still registered with the caching allocator)")


def allocator_clean() -> bool:
    """No capture registered and no pool use count below zero on this rank thread."""
    st = _alloc_state()
    return not st["underway"] and all(v >= 0 for v in st["uc"].values())


def _vclock() -> float:
    return float(getattr(_CUDA_TLS, "clock", 0.0))


def _vclock_advance(ms: float) -> None:
    _CUDA_TLS.clock = _vclock() + float(ms)


GRAPH_COST_FN = None                 # tests: fn(ops, world) -> ms of one replay of a graph with these recorded ops


class Stream:
    _next = [1]

    def __init__(self, device=None, priority=0, _default=False):
        if _default:
            self.id = 0
        else:
            self.id = Stream._next[0]
            Stream._next[0] += 1
        self.cuda_stream = self.id
        self.device = _Device(device) if device is not None else _Device("cuda:0")
        self.priority = priority
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(getattr(other, "id", None))

    def synchronize(self):
        _not_capturing("Stream.synchronize")
        _CudaState.syncs += 1

    def __eq__(self, o):
        return isinstance(o, Stream) and o.id == self.id

    def __hash__(self):
        return hash(("stream", self.id))

    def __repr__(self):
        return f"FakeStream({self.id})"


DEFAULT_STREAM = Stream(_default=True)


def current_stream(device=None):
    return getattr(_CUDA_TLS, "stream", None) or DEFAULT_STREAM


def set_stream(s):
    _CUDA_TLS.stream = s


class StreamContext:
    def __init__(self, s):
        self.s = s
        self.prev = None

    def __enter__(self):
        self.prev = current_stream()
        if self.s is not None:
            set_stream(self.s)
        return self.s

    def __exit__(self, *exc):
        set_stream(self.prev)
        return False


def stream(s):
    return StreamContext(s)


class CUDAGraph:
    REGISTRY: list = []

    def __init__(self):
        self.ops: list = []
        self.captured = False
        self.was_reset = False
        self.replays = 0
        self.capture_mode = None
        self.capture_stream = None
        w = current_world()
        self.rank = getattr(w, "rank", None)
        CUDAGraph.REGISTRY.append(self)

    def replay(self):
        _not_capturing("CUDAGraph.replay")
        if not self.captured or self.was_reset:
            raise RuntimeError("replay of a CUDA graph that holds no captured work")
        w = current_world()
        hook = getattr(w, "replay_hook", None)
        if hook is not None:
            hook(self)                  # an injected launch failure raises BEFORE anything is issued
        for op in self.ops:
            if op[0] == "oneshot":
                op[2]._issue_replayed(op[1])
        _vclock_advance(GRAPH_COST_FN(self.ops, w) if GRAPH_COST_FN is not None else 0.0)
        self.replays += 1

    def reset(self):
        if getattr(self, "capture_ended", False) and getattr(self, "pool_id", None) is not None:
            _cuda_releasePool(0, self.pool_id)          # torch: reset() releases the pool of an ENDED capture only
        self.capture_ended = False
        self.was_reset = True
        self.captured = False
        self.ops = []


class graph:
    """torch.cuda.graph(cuda_graph, pool=None, stream=None, capture_error_mode='global')."""

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        if capture_error_mode not in ("global", "thread_local", "relaxed"):
            raise RuntimeError(f"Unknown capture error mode {capture_error_mode!r}")
        self.g, self.mode = cuda_graph, capture_error_mode
        self.stream = stream if stream is not None else Stream()
        self.pool = pool
        self._ctx = None

    def __enter__(self):
        _cuda_sync()                    # torch.cuda.graph synchronises (and empties the cache) before capturing
        _empty_cache_hook()
        if capturing_graph() is not None:
            raise RuntimeError("nested CUDA graph capture")
        if self.stream.id == 0:
            raise RuntimeError("CUDA graphs must be captured on a non-default stream.")
        self._ctx = StreamContext(self.stream)
        self._ctx.__enter__()
        self.g.pool_id = self.pool if self.pool is not None else graph_pool_handle()
        self.g.capture_ended = False
        _pool_begin(self.g.pool_id)    # torch: beginAllocateToPool BEFORE cudaStreamBeginCapture
        _CUDA_TLS.capture = self.g
        _CUDA_TLS.invalidated = False
        self.g.ops, self.g.capture_mode, self.g.capture_stream = [], self.mode, self.stream
        hook = getattr(current_world(), "capture_hook", None)
        if hook is not None:
            try:
                hook(self.g)            # an injected failure here = cudaStreamBeginCapture failing
            except BaseException:
                _CUDA_TLS.capture = None
                if not STUCK_ON_INVALIDATED[0]:
                    _cuda_endAllocateToPool(0, self.g.pool_id)
                self._ctx.__exit__(None, None, None)
                raise
        return self

    def __exit__(self, et, ev, tb):
        invalidated = getattr(_CUDA_TLS, "invalidated", False)
        _CUDA_TLS.capture = None
        _CUDA_TLS.invalidated = False
        if invalidated:
            # torch's capture_end: cudaStreamEndCapture fails -> AT_CUDA_CHECK raises before endAllocateToPool (the
            # pool stays registered) and torch.cuda.graph.__exit__ skips stream_ctx.__exit__ (the stream stays)
            if not STUCK_ON_INVALIDATED[0]:
                _cuda_endAllocateToPool(0, self.g.pool_id)
            raise RuntimeError("CUDA error: operation failed due to a previous error during capture "
                               "(cudaErrorStreamCaptureInvalidated)")
        _cuda_endAllocateToPool(0, self.g.pool_id)
        self.g.capture_ended = True     # a Python error inside a VALID capture still ends (and instantiates) it
        self._ctx.__exit__(et, ev, tb)
        if et is None:
            self.g.captured = True
        return False


def is_current_stream_capturing():
    return capturing_graph() is not None


class Event:
    """CUDA event on the VIRTUAL GPU clock of the recording thread (graph replays advance it; eager work is free)."""

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        self.t = _vclock()

    def elapsed_time(self, other):
        _not_capturing("Event.elapsed_time")
        return float(other.t - self.t)

    def synchronize(self):
        _not_capturing("Event.synchronize")


def _cuda_sync(device=None):
    _not_capturing("torch.cuda.synchronize")
    _CudaState.syncs += 1


def _empty_cache_hook():
    tm = sys.modules.get("torch")
    fn = getattr(getattr(tm, "cuda", None), "empty_cache", None)
    if fn is not None:
        fn()


# ---------------------------------------------------------------------------------------------------------------------
# build the fake torch module tree
# ---------------------------------------------------------------------------------------------------------------------
class _OpsNS:
    def __init__(self):
        self._C = types.SimpleNamespace()


def make_fake_torch() -> types.ModuleType:
    tm = types.ModuleType("torch")
    tm.__path__ = []
    me = sys.modules[__name__]
    for n in ("Tensor", "Size", "dtype", "device", "Generator", "zeros", "empty", "ones", "full", "zeros_like",
              "empty_like", "ones_like", "tensor", "arange", "randn", "cat", "stack", "where", "isfinite", "equal",
              "matmul", "mm", "no_grad", "inference_mode", "get_default_dtype", "set_default_dtype"):
        setattr(tm, n, getattr(me, n))
    for k, d in _DTYPES.items():
        setattr(tm, "bool" if k == "bool" else k, d)
    tm.float, tm.double, tm.half, tm.int, tm.long = float32, float64, float16, int32, int64
    tm.nn = types.ModuleType("torch.nn")
    tm.nn.__path__ = []
    tm.nn.Module, tm.nn.Parameter, tm.nn.Linear, tm.nn.ModuleDict = Module, Parameter, Linear, ModuleDict
    tm.nn.functional = types.ModuleType("torch.nn.functional")
    tm.nn.functional.linear, tm.nn.functional.pad = linear, pad
    tm.nn.functional.embedding = lambda i, w: w[i]
    tm.cuda = types.ModuleType("torch.cuda")
    tm.cuda.Event = Event
    tm.cuda.synchronize = _cuda_sync
    tm.cuda.is_available = lambda: True
    tm.cuda.is_current_stream_capturing = is_current_stream_capturing
    tm.cuda.Stream, tm.cuda.stream, tm.cuda.current_stream = Stream, stream, current_stream
    tm.cuda.set_stream, tm.cuda.default_stream = set_stream, (lambda device=None: DEFAULT_STREAM)
    tm.cuda.CUDAGraph, tm.cuda.graph = CUDAGraph, graph
    tm.cuda.empty_cache = empty_cache
    tm.cuda.graph_pool_handle = graph_pool_handle
    tm._C = types.SimpleNamespace(_cuda_endAllocateToPool=_cuda_endAllocateToPool,
                                  _cuda_releasePool=_cuda_releasePool)
    tm.cuda.get_device_capability = lambda d=None: (12, 1)
    tm.cuda.current_device = lambda: 0
    tm.ops = _OpsNS()
    tm.distributed = types.ModuleType("torch.distributed")
    tm.Parameter = Parameter
    return tm


# ---------------------------------------------------------------------------------------------------------------------
# N-thread rank emulation
# ---------------------------------------------------------------------------------------------------------------------
class ScheduleMismatch(RuntimeError):
    pass


def timeout_scale() -> float:
    """Rank threads share one process, its GIL and the Mac's cores with whatever else runs there: a rendezvous that waits a few seconds for
    the other ranks must not turn a busy machine into a 'HANG'. $PATCH_TEST_TIMEOUT_SCALE (>= 1, default 1) multiplies every rendezvous
    and rank-thread timeout of this library; a suite runner sets 4 when it runs the suites in parallel."""
    try:
        return max(1.0, float(os.environ.get("PATCH_TEST_TIMEOUT_SCALE", "1")))
    except ValueError:
        return 1.0


class Rendezvous:
    def __init__(self, n: int, timeout: float = 6.0):
        self.n, self.timeout = n, timeout * timeout_scale()
        self.cv = threading.Condition()
        self.slots: dict = {}
        self.hangs: list = []
        self.mismatches: list = []
        self.schedules = [[] for _ in range(n)]

    def exchange(self, rank: int, seq: int, sig, value):
        self.schedules[rank].append(sig)
        deadline = time.monotonic() + self.timeout
        with self.cv:
            slot = self.slots.setdefault(seq, {})
            slot[rank] = (sig, value)
            self.cv.notify_all()
            while len(slot) < self.n:
                left = deadline - time.monotonic()
                if left <= 0:
                    missing = sorted(set(range(self.n)) - set(slot))
                    self.hangs.append((seq, sig, missing))
                    raise TimeoutError(f"HANG: collective #{seq} {sig} - rank(s) {missing} never issued it")
                self.cv.wait(left)
            sigs = {s for s, _ in slot.values()}
            vals = [slot[r][1] for r in range(self.n)]
        if len(sigs) != 1:
            self.mismatches.append((seq, sorted(map(str, sigs))))
            raise ScheduleMismatch(f"collective #{seq}: ranks issued different collectives {sorted(map(str, sigs))}")
        return vals


class FakeOneshot:
    """Patch 012's GB10OneShotAllReduce as its kernel computes: every rank's bf16 input, fp32 sum in rank order
    0..W-1 starting from 0.0, one bf16 rounding; identical bits on every rank."""

    def __init__(self, world, max_elems: int):
        self.world = world
        self.rank, self.world_size, self.max_elems = world.rank, world.size, int(max_elems)
        self.calls = 0
        self.captured = 0
        self.replayed = 0

    def should_use(self, x) -> bool:
        return x.dtype is bfloat16 and x.is_contiguous() and 0 < x.numel() <= self.max_elems

    def __call__(self, x):
        assert self.should_use(x), (x.dtype, tuple(x.shape))
        g = capturing_graph()
        if g is not None:          # captured: recorded, nothing communicated (the output comes from the graph pool)
            self.captured += 1
            g.ops.append(("oneshot", tuple(x.shape), self))
            return Tensor(np.zeros(x._a.shape, dtype=np.uint16), bfloat16, x.device)
        self.calls += 1
        vals = self.world._collective(("oneshot", tuple(x.shape), x.dtype.name), x._a.copy())
        acc = np.zeros(x._a.shape, dtype=np.float32)
        for v in vals:
            acc = (acc + bf16_bits_to_f32(v)).astype(np.float32)
        return Tensor(f32_to_bf16_bits(acc), bfloat16, x.device)

    def _issue_replayed(self, shape):
        """One captured one-shot at a graph replay: the same collective as an eager call of that shape."""
        self.replayed += 1
        self.world._collective(("oneshot", tuple(shape), "bfloat16"), np.zeros(tuple(shape), dtype=np.uint16))


class ThreadWorld:
    kind = "test"

    def __init__(self, rv: Rendezvous, rank: int, size: int, oneshot_max: int | None = 2048, pp_size: int = 1):
        self.rv, self.rank, self.size, self.pp_size = rv, rank, size, pp_size
        self.seq = 0
        self._oneshot = FakeOneshot(self, oneshot_max) if oneshot_max else None
        self.fail_barrier = False
        self.fail_gather_object = False
        self.graph_cost = None          # per-rank overrides of the cpu_check's graph cost model
        self.capture_hook = None        # fn(graph) at capture begin (raise = a failed capture)
        self.replay_hook = None         # fn(graph) before a replay launches (raise = a failed launch)

    def _collective(self, sig, value):
        s = self.seq
        self.seq += 1
        return self.rv.exchange(self.rank, s, sig, value)

    def oneshot(self):
        return self._oneshot

    def all_gather_object(self, obj):
        import copy
        vals = self._collective(("all_gather_object",), copy.deepcopy(obj))
        if self.fail_gather_object:
            raise RuntimeError("injected all_gather_object failure (local, after the collective)")
        return [copy.deepcopy(v) for v in vals]

    def barrier_bounded(self, timeout_s):
        if self.fail_barrier:
            raise RuntimeError("injected monitored_barrier timeout")
        self._collective(("barrier",), None)

    def all_gather_tensor(self, t, dim=-1):
        vals = self._collective(("all_gather", tuple(t.shape), t.dtype.name), np.ascontiguousarray(t._a).copy())
        return Tensor(np.concatenate(vals, axis=dim), t.dtype, t.device)


_TLS = threading.local()


def current_world():
    return getattr(_TLS, "world", None)


def set_current_world(w):
    _TLS.world = w


def run_ranks(n: int, fn, timeout: float = 60.0):
    """fn(rank) on n threads; returns (results, errors). Each thread's exceptions are collected, never swallowed."""
    timeout = timeout * timeout_scale()
    res, err = [None] * n, [None] * n

    def body(r):
        try:
            res[r] = fn(r)
        except BaseException as e:  # noqa: BLE001
            import traceback
            err[r] = (e, traceback.format_exc())

    ths = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout)
    alive = [r for r, t in enumerate(ths) if t.is_alive()]
    if alive:
        raise TimeoutError(f"rank thread(s) {alive} still running after {timeout} s (deadlock)")
    return res, err
