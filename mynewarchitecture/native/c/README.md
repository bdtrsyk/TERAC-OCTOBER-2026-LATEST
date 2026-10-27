# Optional C backend

## Purpose
Packed ternary load/store and sparse eligibility decay kernels.

## ABI
```c
// values packed 2 bits each: 00=+1? mapping documented in kernel
int tcai_packed_validate(const int8_t* values, size_t n);
void tcai_eligibility_decay(float* elig, size_t n, float decay, float floor);
```

## Ownership / lifetime
Caller owns all buffers. No heap allocation inside the C API.

## Thread model
Functions are re-entrant; caller serializes synaptic writes.

## Error handling
Return 0 on success, nonzero error codes otherwise.

## Build
Not required for the Julia reference. If present:
`cc -shared -O3 -o libtcai_c.so eligibility.c`

## Julia interface
Loaded optionally via `ccall` only when `config` enables native packs.
Julia Int8 reference path remains default and behaviorally authoritative.
