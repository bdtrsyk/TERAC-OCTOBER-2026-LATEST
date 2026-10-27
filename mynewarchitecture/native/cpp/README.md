# Optional C++ backend

## Purpose
Sparse ternary graph traversal helpers.

## ABI
C-compatible extern "C" wrappers around C++ sparse maps.

## Ownership / lifetime
Opaque handles created/destroyed by explicit API calls.

## Thread model
One writer per network; readers may be concurrent if frozen.

## Error handling
Exceptions converted to integer status codes at the ABI boundary.

## Build
Optional. Julia reference does not require this library.

## Julia interface
Optional `ccall` bridge; disabled by default.
