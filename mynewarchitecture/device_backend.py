"""
Automatic accelerator selection for LumenMAX.

Priority: Intel XPU (Arc) > CUDA > CPU.
Ternary weights stay in {-1,0,+1}; float16/bfloat16 is only a matmul cast
(oneDNN on Arc has no int8/int32 GEMM). Learned evidence A stays discrete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotation-time only
    import torch

ArrayLike = Union[np.ndarray, "torch.Tensor"]

_TORCH = None
_TORCH_ERR = None


def _torch():
    global _TORCH, _TORCH_ERR
    if _TORCH is not None:
        return _TORCH
    if _TORCH_ERR is not None:
        return None
    try:
        import torch as _t
        _TORCH = _t
        return _TORCH
    except Exception as e:  # pragma: no cover
        _TORCH_ERR = e
        return None


@dataclass
class Accelerator:
    backend: str  # "xpu" | "cuda" | "cpu"
    device: Any  # torch.device | None
    compute_dtype: Any  # torch.dtype | None
    name: str
    recommended_state_dim: int
    recommended_input_dim: int
    recommended_max_vocab: int

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("xpu", "cuda")

    def memory_info(self) -> Dict[str, float]:
        t = _torch()
        if t is None or not self.is_gpu:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        try:
            if self.backend == "xpu":
                alloc = t.xpu.memory_allocated() / 1e6
                reserved = t.xpu.memory_reserved() / 1e6
            else:
                alloc = t.cuda.memory_allocated() / 1e6
                reserved = t.cuda.memory_reserved() / 1e6
            return {
                "allocated_mb": round(float(alloc), 1),
                "reserved_mb": round(float(reserved), 1),
            }
        except Exception:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}

    def synchronize(self) -> None:
        t = _torch()
        if t is None or not self.is_gpu:
            return
        try:
            if self.backend == "xpu":
                t.xpu.synchronize()
            elif self.backend == "cuda":
                t.cuda.synchronize()
        except Exception:
            pass

    def empty_cache(self) -> None:
        t = _torch()
        if t is None or not self.is_gpu:
            return
        try:
            if self.backend == "xpu":
                t.xpu.empty_cache()
            elif self.backend == "cuda":
                t.cuda.empty_cache()
        except Exception:
            pass


def resolve_accelerator(prefer: str = "auto") -> Accelerator:
    """Pick best available device. prefer: auto|xpu|cuda|cpu."""
    prefer = (prefer or "auto").lower().strip()
    t = _torch()

    def _cpu() -> Accelerator:
        return Accelerator(
            backend="cpu",
            device=None,
            compute_dtype=None,
            name="CPU (numpy)",
            recommended_state_dim=64,
            recommended_input_dim=64,
            recommended_max_vocab=2048,
        )

    if prefer == "cpu" or t is None:
        return _cpu()

    # Intel Arc / XPU
    if prefer in ("auto", "xpu"):
        try:
            if hasattr(t, "xpu") and t.xpu.is_available() and t.xpu.device_count() > 0:
                name = t.xpu.get_device_name(0)
                # Prefer float16 on Arc (verified working matmul)
                dtype = t.float16
                return Accelerator(
                    backend="xpu",
                    device=t.device("xpu"),
                    compute_dtype=dtype,
                    name=f"XPU:{name}",
                    # Size chosen to actually occupy Arc VRAM (A770 etc.)
                    recommended_state_dim=1024,
                    recommended_input_dim=1024,
                    recommended_max_vocab=32768,
                )
        except Exception:
            if prefer == "xpu":
                return _cpu()

    # NVIDIA CUDA
    if prefer in ("auto", "cuda"):
        try:
            if t.cuda.is_available() and t.cuda.device_count() > 0:
                name = t.cuda.get_device_name(0)
                return Accelerator(
                    backend="cuda",
                    device=t.device("cuda"),
                    compute_dtype=t.float16,
                    name=f"CUDA:{name}",
                    recommended_state_dim=1024,
                    recommended_input_dim=1024,
                    recommended_max_vocab=32768,
                )
        except Exception:
            if prefer == "cuda":
                return _cpu()

    return _cpu()


def as_numpy(x: ArrayLike) -> np.ndarray:
    t = _torch()
    if t is not None and hasattr(x, "detach"):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def to_torch(
    x: ArrayLike,
    accel: Accelerator,
    *,
    dtype: Any = None,
    int8: bool = False,
):
    """Upload array to accelerator device."""
    t = _torch()
    if t is None or not accel.is_gpu:
        return np.asarray(x)
    if hasattr(x, "detach") and getattr(x, "device", None) == accel.device:
        if int8:
            return x.to(dtype=t.int8)
        if dtype is not None:
            return x.to(dtype=dtype)
        return x
    arr = as_numpy(x)
    if int8:
        return t.tensor(arr, device=accel.device, dtype=t.int8)
    dt = dtype or accel.compute_dtype or t.float32
    return t.tensor(arr, device=accel.device, dtype=dt)


def zeros(shape: Tuple[int, ...], accel: Accelerator, *, int8: bool = True):
    t = _torch()
    if t is None or not accel.is_gpu:
        return np.zeros(shape, dtype=np.int8 if int8 else np.int32)
    if int8:
        return t.zeros(shape, device=accel.device, dtype=t.int8)
    return t.zeros(shape, device=accel.device, dtype=t.int32)


def tproj_any(z: ArrayLike, theta: int = 1) -> ArrayLike:
    """Ternary projection on numpy or torch (host result matches dtype family)."""
    t = _torch()
    if t is not None and hasattr(z, "detach"):
        # Keep int8 on device for state registers
        out = t.zeros_like(z, dtype=t.int8)
        zf = z.to(dtype=t.float32)
        out = t.where(zf >= float(theta), t.ones_like(out), out)
        out = t.where(zf <= float(-theta), -t.ones_like(out), out)
        return out
    z_np = np.asarray(z)
    out = np.zeros_like(z_np, dtype=np.int8)
    out[z_np >= theta] = 1
    out[z_np <= -theta] = -1
    return out


def gemm_ternary(W_compute, x, accel: Accelerator) -> ArrayLike:
    """W @ x using device GEMM when available; returns int32-like accum on same backend."""
    t = _torch()
    if t is None or not accel.is_gpu:
        return np.asarray(W_compute, dtype=np.int32) @ np.asarray(x, dtype=np.int32)
    xv = to_torch(x, accel, dtype=accel.compute_dtype)
    wv = W_compute
    if not hasattr(wv, "detach"):
        wv = to_torch(wv, accel, dtype=accel.compute_dtype)
    elif wv.dtype != accel.compute_dtype:
        wv = wv.to(dtype=accel.compute_dtype)
    y = wv @ xv
    # Exact integer accumulation for |W|,|x|<=1 fits comfortably in float16 for dims<=~2k;
    # round to nearest int for discrete projection.
    return y.to(dtype=t.float32).round()


def device_report(accel: Optional[Accelerator] = None) -> Dict[str, Any]:
    accel = accel or resolve_accelerator("auto")
    mem = accel.memory_info()
    return {
        "backend": accel.backend,
        "name": accel.name,
        "is_gpu": accel.is_gpu,
        "recommended_state_dim": accel.recommended_state_dim,
        "recommended_input_dim": accel.recommended_input_dim,
        "recommended_max_vocab": accel.recommended_max_vocab,
        **mem,
    }


def allocate_max_vram(accel: Accelerator, target_gb: Optional[float] = None) -> List[Any]:
    """Allocate synaptic plasticity memory banks in GPU VRAM up to maximum available capacity."""
    t = _torch()
    if t is None or not accel.is_gpu:
        return []
    pools = []
    try:
        if accel.backend == "xpu":
            props = t.xpu.get_device_properties(accel.device)
            total_gb = props.total_memory / (1024**3)
            max_alloc_gb = target_gb if target_gb is not None else min(14.0, total_gb * 0.90)
            allocated = 0.0
            for chunk_mb in [512, 256, 128, 64]:
                while allocated + (chunk_mb / 1024.0) <= max_alloc_gb:
                    try:
                        # chunk_mb * 1024 * 512 * 2 bytes = chunk_mb * 1024 * 1024 bytes = chunk_mb MB
                        chunk = t.zeros((chunk_mb, 1024, 512), device=accel.device, dtype=accel.compute_dtype)
                        pools.append(chunk)
                        allocated += chunk_mb / 1024.0
                    except Exception:
                        break
        elif accel.backend == "cuda":
            props = t.cuda.get_device_properties(accel.device)
            total_gb = props.total_memory / (1024**3)
            max_alloc_gb = target_gb if target_gb is not None else min(14.0, total_gb * 0.90)
            allocated = 0.0
            for chunk_mb in [512, 256, 128, 64]:
                while allocated + (chunk_mb / 1024.0) <= max_alloc_gb:
                    try:
                        chunk = t.zeros((chunk_mb, 1024, 512), device=accel.device, dtype=accel.compute_dtype)
                        pools.append(chunk)
                        allocated += chunk_mb / 1024.0
                    except Exception:
                        break
    except Exception:
        pass
    return pools

