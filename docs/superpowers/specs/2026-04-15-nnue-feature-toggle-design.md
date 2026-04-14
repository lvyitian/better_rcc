# NNUE Feature Toggle — Spec

## Context

The NNUE neural network evaluation is currently compiled unconditionally, even when not used. Users who want a lean build with only handcrafted evaluation have no way to disable NNUE compilation.

## Goal

Make NNUE an optional compile-time feature via a `nnue` Cargo feature flag.

## Design

### Feature Flag

```toml
# Cargo.toml
[features]
default = []
nnue = []   # zstd and bincode already available as regular deps
```

- Default is **disabled** (`default = []` means no features enabled by default)
- `cargo build` → pure handcrafted evaluation, no NNUE code compiled
- `cargo build --features nnue` → full hybrid NNUE evaluation

### Changes

| File | What changes |
|------|--------------|
| `Cargo.toml` | Add `nnue = []` to `[features]` |
| `src/nn_eval.rs` | Gate `NNUEFeedForward`, `nn_evaluate_or_handcrafted`, `NNOutput`, `Accumulator`, and `InputPlanes` with `#[cfg(feature = "nnue")]`; gate `static NN_NET` lazy init with `#[cfg(feature = "nnue")]` |
| `src/main.rs` | `evaluate()` function: when `nnue` disabled, call `handcrafted_evaluate()` directly; when enabled, call `nn_evaluate_or_handcrafted` as today. The `use crate::nn_eval` import is also gated |

### Evaluation dispatch

- **nnue disabled**: `evaluate()` calls `handcrafted_evaluate()` directly — no NNUE types or functions compiled
- **nnue enabled**: `evaluate()` calls `nn_evaluate_or_handcrafted` which blends NN + handcrafted

### What stays always-compiled

- `src/nnue_input.rs` — only used by `nn_eval.rs`, will not be compiled when feature is off (no orphan rules violated since nothing external references it)

### Constraints

- No runtime feature detection — compile-time only
- `nnue_input.rs` and `nn_eval.rs` must not break when compiled with the feature off (they're just not compiled)

## Success criteria

1. `cargo build` succeeds and produces a binary that uses pure handcrafted evaluation
2. `cargo build --features nnue` succeeds and produces a binary with NNUE hybrid evaluation
3. Both modes pass existing tests
4. No binary size increase for the default (no-nnue) build