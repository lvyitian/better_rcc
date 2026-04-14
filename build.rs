fn main() {
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