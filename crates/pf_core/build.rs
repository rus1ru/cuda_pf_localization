// Links the pf_kernels CUDA library (built by CMake into <pkg>/build/)
// when the `cuda` feature is enabled.
//
// Search order:
//   PF_KERNELS_LIB_DIR (explicit override)
//   <manifest>/../../build            (in-tree cmake build dir)
//   /usr/local/lib, /usr/lib          (installed)

use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    println!("cargo:rustc-link-lib=dylib=pf_kernels");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let pkg_root = manifest
        .parent() // crates/
        .and_then(|p| p.parent()) // package root
        .expect("package root");

    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("PF_KERNELS_LIB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    dirs.push(pkg_root.join("build"));
    dirs.push(PathBuf::from("/usr/local/lib"));
    dirs.push(PathBuf::from("/usr/lib"));

    for d in &dirs {
        if d.join("libpf_kernels.so").exists() {
            println!("cargo:rustc-link-search=native={}", d.display());
            return;
        }
    }
    panic!(
        "libpf_kernels.so not found; build it with:\n  cd {}\n  cmake -B build && cmake --build build\n(searched: {:?})",
        pkg_root.display(),
        dirs
    );
}
