// Increase stack size for deep recursive search and move validation
fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let stack_size = "33554432"; // 32MB

    if target.contains("windows-msvc") {
        // MSVC linker syntax
        println!("cargo:rustc-link-arg=/STACK:{}", stack_size);
    } else {
        // MinGW / GNU ld syntax
        println!("cargo:rustc-link-arg=-Wl,--stack,{}", stack_size);
    }

    // Rerun if TARGET changes
    println!("cargo:rerun-if-env-changed=TARGET");
}
