//! Rust ownership wrapper for the C++/SYCL Arc allocator.

use crate::{Error, Result};
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
struct RawContext { _opaque: [u8; 0] }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawMetrics {
    pub internal_current_bytes: u64,
    pub internal_peak_bytes: u64,
    pub external_current_bytes: u64,
    pub external_peak_bytes: u64,
    pub external_telemetry_available: i32,
}

#[link(name = "tritier_sycl")]
extern "C" {
    fn tt_gpu_create(hard_limit_bytes: u64, error: *mut c_char, error_capacity: usize) -> *mut RawContext;
    fn tt_gpu_destroy(ctx: *mut RawContext);
    fn tt_gpu_alloc(ctx: *mut RawContext, bytes: u64, error: *mut c_char, error_capacity: usize) -> *mut c_void;
    fn tt_gpu_copy_h2d(ctx: *mut RawContext, device_ptr: *mut c_void, host_ptr: *const c_void, bytes: u64, error: *mut c_char, error_capacity: usize) -> i32;
    fn tt_gpu_free(ctx: *mut RawContext, device_ptr: *mut c_void);
    fn tt_gpu_metrics_get(ctx: *mut RawContext) -> RawMetrics;
    fn tt_gpu_device_name(ctx: *mut RawContext) -> *const c_char;
}

fn error_text(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() }
}

#[allow(dead_code)]
pub struct Device { raw: *mut RawContext, pub name: String }
unsafe impl Send for Device {}

#[allow(dead_code)]
impl Device {
    pub fn device_name(&self) -> &str {
        &self.name
    }
    pub fn create(hard_limit: u64) -> Result<Self> {
        let mut error = [0i8; 512];
        let raw = unsafe { tt_gpu_create(hard_limit, error.as_mut_ptr(), error.len()) };
        if raw.is_null() { return Err(Error::Unsupported(format!("SYCL context: {}", error_text(&error)))); }
        let name = unsafe { CStr::from_ptr(tt_gpu_device_name(raw)).to_string_lossy().into_owned() };
        Ok(Self { raw, name })
    }
    pub fn upload(&mut self, host: &[u8]) -> Result<usize> {
        let mut error = [0i8; 512];
        let ptr = unsafe { tt_gpu_alloc(self.raw, host.len() as u64, error.as_mut_ptr(), error.len()) };
        if ptr.is_null() { return Err(Error::Budget(format!("SYCL allocation: {}", error_text(&error)))); }
        let copied = unsafe { tt_gpu_copy_h2d(self.raw, ptr, host.as_ptr().cast(), host.len() as u64, error.as_mut_ptr(), error.len()) };
        if copied == 0 { unsafe { tt_gpu_free(self.raw, ptr) }; return Err(Error::Io(format!("SYCL H2D copy: {}", error_text(&error)))); }
        Ok(ptr as usize)
    }
    pub fn free(&mut self, ptr: usize) { if ptr != 0 { unsafe { tt_gpu_free(self.raw, ptr as *mut c_void) } } }
    pub fn metrics(&self) -> RawMetrics { unsafe { tt_gpu_metrics_get(self.raw) } }
}
impl Drop for Device { fn drop(&mut self) { unsafe { tt_gpu_destroy(self.raw) } } }
