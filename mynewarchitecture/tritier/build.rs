fn main() {
    if std::env::var_os("CARGO_FEATURE_SYCL").is_some() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let default_dir = std::path::PathBuf::from(manifest_dir).join("cpp").join("build");
        let dir = std::env::var("TRITIER_SYCL_LIB_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or(default_dir);
        let dir_str = dir.to_str().expect("valid utf-8 path");
        println!("cargo:rustc-link-search=native={dir_str}");
        println!("cargo:rustc-link-lib=static=tritier_sycl");
        println!("cargo:rustc-link-lib=dxgi");
        println!("cargo:rerun-if-env-changed=TRITIER_SYCL_LIB_DIR");
        println!("cargo:rerun-if-changed={}", dir.join("tritier_sycl.lib").display());
    }
}
