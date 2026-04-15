fn main() {
    // Increase stack size for deep recursive search and move validation
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
    // Rerun this build script if RUSTFLAGS changes (affects compilation)
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    // Rerun if target-cpu changes (affects code generation)
    println!("cargo:rerun-if-env-changed=TARGET_CPU_ARGS");
    // Rerun if any source file changes (ensures fresh build on code changes)
    println!("cargo:rerun-if-changed=src/");
    // Apply -ffast-math and -C target-cpu=native for optimized builds.
    // This is propagated to all crates in the workspace via RUSTFLAGS.
    let current_flags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let fast_math = "-C llvm-args=-ffast-math";
    let target_cpu = "-C target-cpu=native";
    let mut new_flags = Vec::new();
    if !current_flags.contains(fast_math) {
        new_flags.push(fast_math);
    }
    if !current_flags.contains(target_cpu) {
        new_flags.push(target_cpu);
    }
    if !new_flags.is_empty() {
        let combined = if current_flags.is_empty() {
            new_flags.join(" ")
        } else {
            format!("{} {}", current_flags, new_flags.join(" "))
        };
        println!("cargo:rustc-env=RUSTFLAGS={}", combined);
    }
}