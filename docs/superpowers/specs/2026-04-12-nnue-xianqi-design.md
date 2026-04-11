# NNUE for Xiangqi — Design Spec

> Status: APPROVED by user 2026-04-12
> Architecture: Bullet-compliant (1260→512)×2→8 NNUE-Dual

---

## 1. Overview

Replace the convolutional ResNet with NNUE (Efficiently Updatable Neural Networks) for CPU-efficient evaluation with O(1) incremental updates during search. Architecture is **bullet-compliant** — same design as the `bullet` library's `(768→128)×2→1` scaled to Xiangqi dimensions with 8 output buckets.

Train in FP32 via burn, infer in int16 quantized.

---

## 2. Input Encoding

**1260 binary features**: 2 × 90 squares × 7 piece types.

```
index = rank * 9 + file   (0–89)
side_offset = 0 for Red, 630 for Black
global_index = side_offset + index
```

Piece types: King(0), Advisor(1), Elephant(2), Horse(3), Chariot(4), Cannon(5), Pawn(6)

**Dual-perspective inputs** (matching bullet's `dual_perspective()`):
- `stm` (side-to-move): Red at base 0, Black at base 630
- `ntm` (not-side-to-move): Black at base 0, Red at base 630 (board flipped vertically)

**Vertical flip** for NTM: rank y → (9 - y) to mirror the board:
```
ntm_index = (9 - rank) * 9 + file
```

Active features at any position: ~32 (number of pieces on board).

---

## 3. Network Architecture

```
Input [1260] sparse binary
  │
  ▼ FT: [1260] × [1260, 512]^T + [512] → [512]
  │ Column-major, QA=255 quantized
  │ Weights: [Accumulator; 1260] where Accumulator = [i16; 512]
  │ Bias: Accumulator{512}
  │
  ▼ SCReLU: clamp(x, 0, QA)² → [512] (returns i32)
  │
  ├──► stm_acc:   [512] (side-to-move perspective)
  │
  └──► ntm_acc:   [512] (not-side-to-move perspective)
              │
         Concatenate [1024]
              │
              ▼
         Output: [1024] × 8 + [8]
              │ QB=64 quantized
              ▼
         Select bucket by non-king piece count
              │
              ▼
         ((Σ(SCReLU * out_w) / QA) + bias) * SCALE / (QA * QB)
              │
              ▼
         tanh × SCALE → scalar ∈ [-400, 400]
```

### Parameter Count

| Layer | Shape | Weights | Biases |
|-------|-------|---------|--------|
| FeatureTransformer | [1260, 512] | 645,120 | 512 |
| Output | [1024, 8] | 8,192 | 8 |
| **Total** | | **653,312** | **520** |

**~654K parameters**

---

## 4. Activation Functions

### SCReLU (Square-Clipped ReLU)

```rust
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y  // squared
}
```

Range: [0, 255²] = [0, 65025] (stored as i32).

### SIMD Optimization

Computing `(v*v)*w` in 16-bit SIMD risks overflow. Bullet's approach: use `_mm256_madd_epi16` which computes `(a*b + c)` as a single instruction. The trick is to compute `(v*w)*v` instead of `(v*v)*w` to avoid intermediate overflow.

---

## 5. Quantization

| Parameter | Value |
|-----------|-------|
| QA | 255 |
| QB | 64 |
| SCALE | 400 |

**Quantization in save format** (matching bullet's `save_format`):
```
l0w: round().quantise::<i16>(QA)    // FT weights, QA quantized
l0b: round().quantise::<i16>(QA)    // FT bias, QA quantized
l1w: round().quantise::<i16>(QB)    // Output weights, QB quantized
l1b: round().quantise::<i16>(QA*QB) // Output bias, QA*QB quantized
```

**Final output formula** (matching bullet's `evaluate`):
```rust
// Sum SCReLU-weighted outputs for both accumulators
output = Σ(screlu(stm_acc[i]) * out_w[i]) + Σ(screlu(ntm_acc[i]) * out_w[512+i]);

// Reduce quantization: QA²·QB → QA·QB
output /= QA;          // divide by QA (255)

// Add bias (already QA·QB quantized)
output += i32::from(output_bias);

// Apply eval scale
output *= SCALE;       // 400

// Remove quantization: QA·QB → 1
output /= QA * QB;     // 255 * 64 = 16320
```

---

## 6. Incremental Updates

**Core property**: FT is purely linear (no activation). SCReLU is applied AFTER accumulation.

### Accumulator State

```rust
#[repr(C, align(64))]
struct Accumulator {
    vals: [i16; 512],
}

struct NNUEState {
    stm_acc: Accumulator,   // side-to-move
    ntm_acc: Accumulator,   // not-side-to-move
    valid: bool,
}
```

### Add/Remove Feature

```rust
fn add_feature(acc: &mut Accumulator, feature_idx: usize, ft_weights: &[Accumulator; 1260]) {
    for (i, d) in acc.vals.iter_mut().zip(&ft_weights[feature_idx].vals) {
        *i = i.saturating_add(*d);  // saturating to avoid overflow
    }
}

fn remove_feature(acc: &mut Accumulator, feature_idx: usize, ft_weights: &[Accumulator; 1260]) {
    for (i, d) in acc.vals.iter_mut().zip(&ft_weights[feature_idx].vals) {
        *i = i.saturating_sub(*d);  // saturating to avoid overflow
    }
}
```

### On Move

- **Quiet move**: remove old feature at src, add new feature at dst
- **Capture**: additionally remove captured piece's feature at dst

### When to Refresh

- First evaluation (full FT forward pass)
- King moves (position changes fundamentally)
- Loading position from history

---

## 7. Bucket Selection

8 buckets based on non-king piece count (0–30):

```rust
fn bucket_index(non_king_count: u8) -> usize {
    ((non_king_count as usize).saturating_sub(2) / 4).min(7)
}
```

| count | bucket |
|-------|--------|
| 0–3 | 0 |
| 4–7 | 1 |
| 8–11 | 2 |
| 12–15 | 3 |
| 16–19 | 4 |
| 20–23 | 5 |
| 24–27 | 6 |
| 28–30 | 7 |

---

## 8. Training (burn)

**Framework**: burn + burn-ndarray + burn-autodiff (FP32)

**Forward chain**:
```
input [1260]
  ↓ ft: 1260→512, bias
  ↓ SCReLU (clamp 0, 255)²
  ↓ split → stm_acc[512], ntm_acc[512]
  ↓ concat [1024]
  ↓ output: 1024→8
  ↓ select bucket by non_king_count
  ↓ formula above
  ↓ tanh × 400
```

**Loss**: MSE on bucket-selected value normalized by 400.

**Optimizer**: AdamW with weight clipping to [-1.98, 1.98] (matching bullet's default).

---

## 9. Files

| File | Responsibility |
|------|---------------|
| `src/nnue_input.rs` | `NNInputPlanes` 1260-dim encoding, dual-perspective feature mapping |
| `src/nnue_eval.rs` | `NNUEFeedForward` (plain ndarray, int16), `NNUEFeedForwardBurn` (burn) |
| `src/board.rs` | Accumulator state, update/undo, refresh |
| `src/nn_train.rs` | `TrainingSample.non_king_count`, forward chain |
| `src/main.rs` | Wire evaluation |

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 512-unit accumulator | Larger than chess's 128, per user request. Maintains expressiveness. |
| 1260-dim input | 7 types × 90 squares × 2 sides. Minimal, no per-piece identity. |
| No hidden layer | Matches bullet. Expressiveness is in the wide FT, not hidden layers. |
| SCReLU (clamp²) | Non-linear activation. Returns i32 to avoid overflow of 255². |
| Column-major weights | Matches bullet's `#[repr(C)]` + `align(64)` for SIMD. |
| QA=255, QB=64 | Matches bullet's NNUE reference. |
| 8 buckets | Standard PSQTBuckets. Bucket formula matches bullet's `(count-2)/4`. |
| Vertical flip (rank 0↔9) | Correct Xiangqi mirror. Chess uses horizontal flip (file 0↔7). |
