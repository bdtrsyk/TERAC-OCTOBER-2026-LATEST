#pragma once

#include <cstddef>
#include <cstdint>

// Stable C ABI. Ownership is explicit: a successful allocation belongs to the
// caller and must be released with tt_gpu_free using the same context.
extern "C" {
struct tt_gpu_context;
struct tt_gpu_metrics {
  std::uint64_t internal_current_bytes;
  std::uint64_t internal_peak_bytes;
  std::uint64_t external_current_bytes;
  std::uint64_t external_peak_bytes;
  int external_telemetry_available;
};

tt_gpu_context* tt_gpu_create(std::uint64_t hard_limit_bytes, char* error, std::size_t error_capacity);
void tt_gpu_destroy(tt_gpu_context* ctx);
void* tt_gpu_alloc(tt_gpu_context* ctx, std::uint64_t bytes, char* error, std::size_t error_capacity);
int tt_gpu_copy_h2d(tt_gpu_context* ctx, void* device_ptr, const void* host_ptr, std::uint64_t bytes, char* error, std::size_t error_capacity);
void tt_gpu_free(tt_gpu_context* ctx, void* device_ptr);
tt_gpu_metrics tt_gpu_metrics_get(tt_gpu_context* ctx);
const char* tt_gpu_device_name(tt_gpu_context* ctx);
}
