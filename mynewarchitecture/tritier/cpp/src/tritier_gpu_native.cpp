#include "tritier_gpu.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dxgi.lib")

namespace {

void set_error(char* out, std::size_t cap, const std::string& message) {
  if (out == nullptr || cap == 0) return;
  std::strncpy(out, message.c_str(), cap - 1);
  out[cap - 1] = '\0';
}

// OpenCL basic types & constants
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef size_t cl_size_t;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_TRUE 1
#define CL_FALSE 0

// Function pointer signatures for OpenCL functions
typedef cl_int (WINAPI *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (WINAPI *PFN_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (WINAPI *PFN_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (WINAPI *PFN_clCreateContext)(const void*, cl_uint, const cl_device_id*, void (WINAPI*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (WINAPI *PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (WINAPI *PFN_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_int (WINAPI *PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (WINAPI *PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (WINAPI *PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (WINAPI *PFN_clReleaseContext)(cl_context);
typedef cl_int (WINAPI *PFN_clFinish)(cl_command_queue);

struct OpenCLBindings {
  HMODULE hModule = nullptr;
  PFN_clGetPlatformIDs clGetPlatformIDs = nullptr;
  PFN_clGetDeviceIDs clGetDeviceIDs = nullptr;
  PFN_clGetDeviceInfo clGetDeviceInfo = nullptr;
  PFN_clCreateContext clCreateContext = nullptr;
  PFN_clCreateCommandQueue clCreateCommandQueue = nullptr;
  PFN_clCreateBuffer clCreateBuffer = nullptr;
  PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
  PFN_clReleaseMemObject clReleaseMemObject = nullptr;
  PFN_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
  PFN_clReleaseContext clReleaseContext = nullptr;
  PFN_clFinish clFinish = nullptr;

  bool load() {
    if (hModule) return true;
    hModule = LoadLibraryA("OpenCL.dll");
    if (!hModule) return false;
    clGetPlatformIDs = (PFN_clGetPlatformIDs)GetProcAddress(hModule, "clGetPlatformIDs");
    clGetDeviceIDs = (PFN_clGetDeviceIDs)GetProcAddress(hModule, "clGetDeviceIDs");
    clGetDeviceInfo = (PFN_clGetDeviceInfo)GetProcAddress(hModule, "clGetDeviceInfo");
    clCreateContext = (PFN_clCreateContext)GetProcAddress(hModule, "clCreateContext");
    clCreateCommandQueue = (PFN_clCreateCommandQueue)GetProcAddress(hModule, "clCreateCommandQueue");
    clCreateBuffer = (PFN_clCreateBuffer)GetProcAddress(hModule, "clCreateBuffer");
    clEnqueueWriteBuffer = (PFN_clEnqueueWriteBuffer)GetProcAddress(hModule, "clEnqueueWriteBuffer");
    clReleaseMemObject = (PFN_clReleaseMemObject)GetProcAddress(hModule, "clReleaseMemObject");
    clReleaseCommandQueue = (PFN_clReleaseCommandQueue)GetProcAddress(hModule, "clReleaseCommandQueue");
    clReleaseContext = (PFN_clReleaseContext)GetProcAddress(hModule, "clReleaseContext");
    clFinish = (PFN_clFinish)GetProcAddress(hModule, "clFinish");
    return clGetPlatformIDs && clGetDeviceIDs && clCreateContext && clCreateBuffer && clReleaseMemObject;
  }
};

static OpenCLBindings g_cl;

} // namespace

struct tt_gpu_context {
  cl_context cl_ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_device_id device = nullptr;
  IDXGIFactory4* dxgi_factory = nullptr;
  IDXGIAdapter3* dxgi_adapter = nullptr;

  std::uint64_t hard_limit = 0;
  std::uint64_t current = 0;
  std::uint64_t peak = 0;
  std::uint64_t ext_current = 0;
  std::uint64_t ext_peak = 0;

  std::unordered_map<void*, std::uint64_t> allocations;
  std::string name = "Unknown GPU";
  std::mutex mutex;

  void sample_dxgi() {
    if (dxgi_adapter) {
      DXGI_QUERY_VIDEO_MEMORY_INFO info;
      if (SUCCEEDED(dxgi_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        ext_current = info.CurrentUsage;
        ext_peak = std::max(ext_peak, ext_current);
      }
    }
  }
};

extern "C" tt_gpu_context* tt_gpu_create(std::uint64_t hard_limit_bytes, char* error, std::size_t error_capacity) {
  if (hard_limit_bytes > 500000000ULL) {
    set_error(error, error_capacity, "hard limit must stay strictly below or equal to 500,000,000 bytes");
    return nullptr;
  }

  if (!g_cl.load()) {
    set_error(error, error_capacity, "Failed to load OpenCL.dll");
    return nullptr;
  }

  // Find GPU platform & device
  cl_uint num_platforms = 0;
  g_cl.clGetPlatformIDs(0, nullptr, &num_platforms);
  if (num_platforms == 0) {
    set_error(error, error_capacity, "No OpenCL platforms found");
    return nullptr;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  g_cl.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  cl_device_id chosen_device = nullptr;
  for (auto p : platforms) {
    cl_uint num_dev = 0;
    if (g_cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_dev) == CL_SUCCESS && num_dev > 0) {
      std::vector<cl_device_id> devices(num_dev);
      g_cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_dev, devices.data(), nullptr);
      chosen_device = devices[0];
      break;
    }
  }

  if (!chosen_device) {
    set_error(error, error_capacity, "No GPU device found on any OpenCL platform");
    return nullptr;
  }

  cl_int err = 0;
  cl_context ctx = g_cl.clCreateContext(nullptr, 1, &chosen_device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS || !ctx) {
    set_error(error, error_capacity, "Failed to create OpenCL context: code " + std::to_string(err));
    return nullptr;
  }

  cl_command_queue queue = g_cl.clCreateCommandQueue(ctx, chosen_device, 0, &err);
  if (err != CL_SUCCESS || !queue) {
    g_cl.clReleaseContext(ctx);
    set_error(error, error_capacity, "Failed to create OpenCL command queue");
    return nullptr;
  }

  char dev_name[256] = {0};
  g_cl.clGetDeviceInfo(chosen_device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);

  auto* result = new tt_gpu_context();
  result->cl_ctx = ctx;
  result->queue = queue;
  result->device = chosen_device;
  result->hard_limit = hard_limit_bytes;
  result->name = dev_name[0] ? dev_name : "Intel Arc Graphics";

  // Setup DXGI telemetry for the corresponding GPU
  IDXGIFactory4* pFactory = nullptr;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)))) {
    result->dxgi_factory = pFactory;
    IDXGIAdapter1* pAdapter = nullptr;
    for (UINT i = 0; pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 desc;
      pAdapter->GetDesc1(&desc);
      IDXGIAdapter3* pAdapter3 = nullptr;
      if (SUCCEEDED(pAdapter->QueryInterface(IID_PPV_ARGS(&pAdapter3)))) {
        // Pick the first non-software adapter (Intel Arc A770)
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
          result->dxgi_adapter = pAdapter3;
          pAdapter->Release();
          break;
        }
        pAdapter3->Release();
      }
      pAdapter->Release();
    }
  }

  result->sample_dxgi();
  return result;
}

extern "C" void tt_gpu_destroy(tt_gpu_context* ctx) {
  if (!ctx) return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  for (const auto& a : ctx->allocations) {
    if (a.first) g_cl.clReleaseMemObject((cl_mem)a.first);
  }
  ctx->allocations.clear();
  if (ctx->queue) g_cl.clReleaseCommandQueue(ctx->queue);
  if (ctx->cl_ctx) g_cl.clReleaseContext(ctx->cl_ctx);
  if (ctx->dxgi_adapter) ctx->dxgi_adapter->Release();
  if (ctx->dxgi_factory) ctx->dxgi_factory->Release();
  delete ctx;
}

extern "C" void* tt_gpu_alloc(tt_gpu_context* ctx, std::uint64_t bytes, char* error, std::size_t error_capacity) {
  if (!ctx || bytes == 0) {
    set_error(error, error_capacity, "invalid allocation parameters");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);

  // Authoritative hard limit accounting (strictly < 500,000,000 bytes)
  if (bytes > ctx->hard_limit || ctx->current > ctx->hard_limit - bytes) {
    set_error(error, error_capacity, "TriTier GPU hard budget (" + std::to_string(ctx->hard_limit) + " bytes) would be exceeded");
    return nullptr;
  }

  cl_int err = 0;
  cl_mem buf = g_cl.clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, static_cast<size_t>(bytes), nullptr, &err);
  if (err != CL_SUCCESS || !buf) {
    set_error(error, error_capacity, "clCreateBuffer failed: code " + std::to_string(err));
    return nullptr;
  }

  ctx->allocations.emplace((void*)buf, bytes);
  ctx->current += bytes;
  ctx->peak = std::max(ctx->peak, ctx->current);
  ctx->sample_dxgi();
  return (void*)buf;
}

extern "C" int tt_gpu_copy_h2d(tt_gpu_context* ctx, void* device_ptr, const void* host_ptr, std::uint64_t bytes, char* error, std::size_t error_capacity) {
  if (!ctx || !device_ptr || !host_ptr) {
    set_error(error, error_capacity, "invalid copy arguments");
    return 0;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  auto it = ctx->allocations.find(device_ptr);
  if (it == ctx->allocations.end() || bytes > it->second) {
    set_error(error, error_capacity, "copy exceeds owned device allocation");
    return 0;
  }

  cl_int err = g_cl.clEnqueueWriteBuffer(ctx->queue, (cl_mem)device_ptr, CL_TRUE, 0, static_cast<size_t>(bytes), host_ptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    set_error(error, error_capacity, "clEnqueueWriteBuffer failed: code " + std::to_string(err));
    return 0;
  }
  return 1;
}

extern "C" void tt_gpu_free(tt_gpu_context* ctx, void* device_ptr) {
  if (!ctx || !device_ptr) return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  auto it = ctx->allocations.find(device_ptr);
  if (it == ctx->allocations.end()) return;

  g_cl.clReleaseMemObject((cl_mem)device_ptr);
  ctx->current -= it->second;
  ctx->allocations.erase(it);
  ctx->sample_dxgi();
}

extern "C" tt_gpu_metrics tt_gpu_metrics_get(tt_gpu_context* ctx) {
  if (!ctx) return {};
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->sample_dxgi();
  return {
    ctx->current,
    ctx->peak,
    ctx->ext_current,
    ctx->ext_peak,
    ctx->dxgi_adapter != nullptr ? 1 : 0
  };
}

extern "C" const char* tt_gpu_device_name(tt_gpu_context* ctx) {
  return ctx ? ctx->name.c_str() : "unavailable";
}
