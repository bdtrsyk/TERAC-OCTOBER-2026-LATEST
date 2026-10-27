# Optional CUDA backend

## Purpose
Batch eligibility decay / sparse gather on GPU for large synapse counts.

## Hard constraint
GPU buffers store Int8 ternary values (or packed ternary). They must NOT
silently become Float32 trainable weights.

## ABI
CUDA kernels launched through a thin C API.

## Ownership / lifetime
Device buffers allocated/freed by explicit API; host owns the model of record.

## Thread model
One CUDA stream per episode by default.

## Error handling
`cudaError_t` surfaced to Julia as exceptions.

## Build
Optional (`nvcc`). CPU reference always available via `--cpu`.

## Julia interface
Enabled only with `--gpu` and a built library; otherwise CPU path runs.
