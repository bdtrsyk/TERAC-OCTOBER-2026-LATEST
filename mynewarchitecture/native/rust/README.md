# Optional Rust backend

## Purpose
Safe packed ternary bitset and SIMD-friendly eligibility updates.

## ABI
`#[no_mangle] pub extern "C"` functions with plain C types.

## Ownership / lifetime
Rust allocates only behind explicit create/destroy handles.

## Thread model
Send+Sync only for immutable snapshots.

## Error handling
Integer status codes; no panics across FFI.

## Build
`cargo build --release` when used. Not required for Julia CPU mode.

## Julia interface
Optional; Julia Int8 synapses remain the reference implementation.
