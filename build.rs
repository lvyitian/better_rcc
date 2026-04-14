fn main() {
    // Apply -ffast-math to enable aggressive floating-point optimizations.
    // This is propagated to all crates in the workspace via RUSTFLAGS.
    let current_flags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let fast_math = "-C llvm-args=-ffast-math";
    if !current_flags.contains(fast_math) {
        let new_flags = if current_flags.is_empty() {
            fast_math.to_string()
        } else {
            format!("{} {}", current_flags, fast_math)
        };
        println!("cargo:rustc-env=RUSTFLAGS={}", new_flags);
    }
}