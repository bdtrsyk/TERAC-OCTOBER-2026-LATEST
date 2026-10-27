#include "tritier_gpu.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
void set_error(char* out, std::size_t cap, const std::string& message) {
  if (out == nullptr || cap == 0) return;
  std::strncpy(out, message.c_str(), cap - 1);
  out[cap - 1] = '\0';
}
}

struct tt_gpu_context {
  sycl::queue queue;
  std::uint64_t hard_limit;
  std::uint64_t current = 0;
  std::uint64_t peak = 0;
  std::unordered_map<void*, std::uint64_t> allocations;
  std::string name;
  std::mutex mutex;

  explicit tt_gpu_context(std::uint64_t limit)
      : queue(sycl::gpu_selector_v), hard_limit(limit),
        name(queue.get_device().get_info<sycl::info::device::name>()) {}
};

extern "C" tt_gpu_context* tt_gpu_create(std::uint64_t hard_limit_bytes, char* error, std::size_t error_capacity) {
  if (hard_limit_bytes > 500000000ULL) {
    set_error(error, error_capacity, "hard limit must stay strictly below or equal to 500,000,000 bytes");
    return nullptr;
  }
  try { return new tt_gpu_context(hard_limit_bytes); }
  catch (const std::exception& e) { set_error(error, error_capacity, e.what()); return nullptr; }
}

extern "C" void tt_gpu_destroy(tt_gpu_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  for (const auto& allocation : ctx->allocations) sycl::free(allocation.first, ctx->queue);
  ctx->allocations.clear();
  delete ctx;
}

extern "C" void* tt_gpu_alloc(tt_gpu_context* ctx, std::uint64_t bytes, char* error, std::size_t error_capacity) {
  if (!ctx || bytes == 0) { set_error(error, error_capacity, "invalid allocation"); return nullptr; }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  // This is the authoritative pre-allocation guard. It accounts for every
  // allocation submitted through this ABI, including staging/workspace buffers.
  if (bytes > ctx->hard_limit || ctx->current > ctx->hard_limit - bytes) {
    set_error(error, error_capacity, "TriTier GPU hard budget would be exceeded"); return nullptr;
  }
  try {
    void* ptr = sycl::malloc_device(static_cast<std::size_t>(bytes), ctx->queue);
    if (!ptr) { set_error(error, error_capacity, "SYCL device allocation failed"); return nullptr; }
    ctx->allocations.emplace(ptr, bytes);
    ctx->current += bytes;
    ctx->peak = std::max(ctx->peak, ctx->current);
    return ptr;
  } catch (const std::exception& e) { set_error(error, error_capacity, e.what()); return nullptr; }
}

extern "C" int tt_gpu_copy_h2d(tt_gpu_context* ctx, void* device_ptr, const void* host_ptr, std::uint64_t bytes, char* error, std::size_t error_capacity) {
  if (!ctx || !device_ptr || !host_ptr) { set_error(error, error_capacity, "invalid copy arguments"); return 0; }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  const auto it = ctx->allocations.find(device_ptr);
  if (it == ctx->allocations.end() || bytes > it->second) { set_error(error, error_capacity, "copy exceeds owned device block"); return 0; }
  try { ctx->queue.memcpy(device_ptr, host_ptr, static_cast<std::size_t>(bytes)).wait(); return 1; }
  catch (const std::exception& e) { set_error(error, error_capacity, e.what()); return 0; }
}

extern "C" void tt_gpu_free(tt_gpu_context* ctx, void* device_ptr) {
  if (!ctx || !device_ptr) return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  const auto it = ctx->allocations.find(device_ptr);
  if (it == ctx->allocations.end()) return;
  ctx->current -= it->second;
  sycl::free(device_ptr, ctx->queue);
  ctx->allocations.erase(it);
}

extern "C" tt_gpu_metrics tt_gpu_metrics_get(tt_gpu_context* ctx) {
  if (!ctx) return {};
  std::lock_guard<std::mutex> lock(ctx->mutex);
  // SYCL exposes allocation truth owned by this process through this allocator.
  // Driver-wide telemetry is platform-specific; false is explicit rather than fake.
  return {ctx->current, ctx->peak, 0, 0, 0};
}
extern "C" const char* tt_gpu_device_name(tt_gpu_context* ctx) { return ctx ? ctx->name.c_str() : "unavailable"; }
