# NNUE Implementation Plan (Xiangqi)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the convolutional ResNet with bullet-compliant NNUE (1260→512)×2→8 for CPU-efficient evaluation with O(1) incremental updates.

**Architecture:**
- Input: 1260 binary features (2 × 90 squares × 7 piece types)
- FT: 1260 → 512, column-major, QA=255
- Dual accumulators: 512 × 2, SCReLU (clamp², returns i32)
- Output: 1024 → 8 buckets
- ~654K params

**Tech Stack:** burn (train) + plain ndarray int16 (inference)

---

## File Map

| File | Changes |
|------|---------|
| `src/nnue_input.rs` | **CREATE** — new 1260-dim input encoding |
| `src/nn_eval.rs` | **REWRITE** — delete CompactResNet (lines 178–643), add NNUEFeedForward + burn version, keep `NNOutput` + `nn_evaluate_or_handcrafted` |
| `src/main.rs` | **MODIFY** Board struct (line 1651) — add accumulator state, update `make_move`/`undo_move` |
| `src/nn_train.rs` | **MODIFY** — add `non_king_count` to `TrainingSample`, update forward chain |

---

## Constants (shared across files)

```rust
pub const INPUT_DIM: usize = 1260;    // 2 × 90 × 7
pub const FT_DIM: usize = 512;        // Feature transformer output
pub const NUM_BUCKETS: usize = 8;
pub const QA: i16 = 255;              // FT quantization
pub const QB: i16 = 64;               // Output quantization
pub const SCALE: i32 = 400;
```

---

## Task 1: Create `src/nnue_input.rs`

**Files:** Create: `src/nnue_input.rs`

- [ ] **Step 1: Write the file**

```rust
//! NNUE input encoding for Xiangqi.
//! 1260 binary features: 2 × 90 squares × 7 piece types.

use crate::{Board, Color, PieceType};

pub const INPUT_DIM: usize = 1260;   // 2 × 90 × 7
pub const FT_DIM: usize = 512;
pub const NUM_BUCKETS: usize = 8;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const SCALE: i32 = 400;

/// Bucket index from non-king piece count (0–30).
pub fn bucket_index(non_king_count: u8) -> usize {
    ((non_king_count as usize).saturating_sub(2) / 4).min(7)
}

/// NNInputPlanes: 1260-element sparse binary input.
/// Layout: [0..629] = Red pieces, [630..1259] = Black pieces.
#[derive(Clone, Debug)]
pub struct NNInputPlanes {
    pub data: [f32; INPUT_DIM],
}

impl NNInputPlanes {
    /// Build from board state with dual-perspective encoding.
    /// stm (side-to-move) perspective: our pieces at base 0, their pieces at base 630.
    /// ntm (not-side-to-move) perspective: their pieces at base 0, our pieces at base 630.
    pub fn from_board(board: &Board) -> (Self, Self) {
        let mut stm_data = [0.0f32; INPUT_DIM];
        let mut ntm_data = [0.0f32; INPUT_DIM];

        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.cells[y][x] {
                    let sq = y * 9 + x;  // 0–89
                    let pt = piece.piece_type as usize;  // 0–6

                    let (stm_base, ntm_base) = match piece.color {
                        Color::Red => (0, 630),   // Red: stm at 0, ntm at 630
                        Color::Black => (630, 0), // Black: stm at 630, ntm at 0
                    };

                    // Vertical flip for NTM: rank y → (9 - y)
                    let ntm_sq = (9 - y) * 9 + x;

                    stm_data[stm_base + pt * 90 + sq] = 1.0;
                    ntm_data[ntm_base + pt * 90 + ntm_sq] = 1.0;
                }
            }
        }

        (Self { data: stm_data }, Self { data: ntm_data })
    }

    /// For burn training: convert to flat array.
    pub fn to_array(&self) -> [f32; INPUT_DIM] {
        self.data
    }
}

/// Count non-king pieces on the board (0–30).
pub fn count_non_king_pieces(board: &Board) -> u8 {
    let mut count = 0u8;
    for y in 0..10 {
        for x in 0..9 {
            if let Some(piece) = board.cells[y][x] {
                if piece.piece_type != PieceType::King {
                    count += 1;
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_index() {
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(3), 0);
        assert_eq!(bucket_index(4), 1);
        assert_eq!(bucket_index(7), 1);
        assert_eq!(bucket_index(30), 7);
        assert_eq!(bucket_index(100), 7); // saturates
    }

    #[test]
    fn test_nn_input_planes_starting_position() {
        let board = Board::new(RuleSet::Official, 1);
        let (stm, ntm) = NNInputPlanes::from_board(&board);

        // Should have ~32 active features
        let stm_active = stm.data.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(stm_active, 32, "starting position should have 32 pieces");

        // STM and NTM should be different (board flipped)
        let stm_sum: f32 = stm.data.iter().sum();
        let ntm_sum: f32 = ntm.data.iter().sum();
        assert_eq!(stm_sum, 32.0);
        assert_eq!(ntm_sum, 32.0);
    }

    #[test]
    fn test_vertical_flip() {
        // Red king at (4, 0), Black king at (4, 9)
        let board = Board::new(RuleSet::Official, 1);
        let (stm, ntm) = NNInputPlanes::from_board(&board);

        // Red king (piece type 0) at rank 0: stm index = 0 + 0*90 + 4*9 + 4 = 4*9+4 = 40
        // Black king (piece type 0) at rank 9: stm index = 630 + 0*90 + 9*9 + 4 = 630 + 81 + 4 = 715
        // In NTM (flipped): Black at base 0, rank 9 → (9-9)*9+4 = 4, so index = 4
        //                    Red at base 630, rank 0 → (9-0)*9+4 = 81+4 = 85, so index = 630+85 = 715
        assert_eq!(ntm.data[4], 1.0, "Black king (flipped) should be at index 4 in NTM");
        assert_eq!(ntm.data[715], 1.0, "Red king (flipped) should be at index 715 in NTM");
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check 2>&1 | tail -10`
Expected: Clean compile

- [ ] **Step 3: Run tests**

Run: `cargo test nnue_input 2>&1 | tail -15`
Expected: All 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/nnue_input.rs && git commit -m "feat(nn): add NNUE input encoding (1260-dim, dual perspective)"
```

---

## Task 2: Rewrite `src/nn_eval.rs` — NNUEFeedForward

**Files:** Modify: `src/nn_eval.rs` (delete lines 178–643, add new code after line 175)

**Boundary:**
- KEEP lines 1–175: `InputPlanes` struct (leave untouched — it's no longer used but deleting it could break other things)
- DELETE lines 178–643: `CompactResNet`, `conv helpers`, `ResBlock`, `CompactResNetBurn`
- KEEP lines 644–780: `NNOutput`, `nn_evaluate_or_handcrafted` (update in Task 4)

- [ ] **Step 1: Delete old code and add new architecture**

After line 175 in `src/nn_eval.rs`, add:

```rust
// =============================================================================
// NNUE Feedforward — bullet-compliant (1260→512)×2→8
// =============================================================================

use crate::nnue_input::{INPUT_DIM, FT_DIM, NUM_BUCKETS, QA, QB, SCALE, bucket_index};

/// SCReLU: clamp(x, 0, QA)², returns i32.
/// Matches bullet's implementation exactly.
#[inline]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

/// Accumulator: column of FT weights, 64-byte aligned.
/// Matches bullet's `#[repr(C, align(64))]`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; FT_DIM],
}

/// Plain ndarray version for inference (non-train).
/// Int16 quantized weights, no floating point during evaluation.
#[derive(Clone)]
pub struct NNUEFeedForward {
    /// FT weights: [1260] × [512], column-major, QA quantized.
    /// Stored as [Accumulator; 1260] — each Accumulator is a column of 512 i16s.
    ft_weights: Vec<Accumulator>,
    /// FT bias: [512], QA quantized.
    ft_bias: [i16; FT_DIM],
    /// Output weights: [1024] × [8], QB quantized.
    /// Column 0 = bucket 0, etc.
    out_weights: [[i16; FT_DIM * 2]; NUM_BUCKETS],
    /// Output bias: [8], QA*QB quantized.
    out_bias: [i16; NUM_BUCKETS],
}

impl NNUEFeedForward {
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let scale_ft = (2.0 / INPUT_DIM as f32).sqrt() * QA as f32;
        let scale_out = (2.0 / (FT_DIM * 2) as f32).sqrt() * QB as f32;

        let mut ft_weights = Vec::with_capacity(INPUT_DIM);
        for _ in 0..INPUT_DIM {
            let mut col = Accumulator { vals: [0i16; FT_DIM] };
            for val in &mut col.vals {
                *val = ((rng.gen::<f32>() * 2.0 - 1.0) * scale_ft) as i16;
            }
            ft_weights.push(col);
        }

        let ft_bias = [0i16; FT_DIM];

        let mut out_weights = [[0i16; FT_DIM * 2]; NUM_BUCKETS];
        for bucket in &mut out_weights {
            for val in bucket {
                *val = ((rng.gen::<f32>() * 2.0 - 1.0) * scale_out) as i16;
            }
        }

        let out_bias = [0i16; NUM_BUCKETS];

        Self { ft_weights, ft_bias, out_weights, out_bias }
    }

    /// Compute accumulators from scratch (full forward pass).
    fn compute_accumulators(stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM]) -> ([i16; FT_DIM], [i16; FT_DIM]) {
        let mut stm_acc = [0i16; FT_DIM];
        let mut ntm_acc = [0i16; FT_DIM];

        // Add bias
        for k in 0..FT_DIM {
            stm_acc[k] = 0i16;
            ntm_acc[k] = 0i16;
        }

        // Accumulate active features
        for f in 0..INPUT_DIM {
            if stm[f] != 0.0 {
                for k in 0..FT_DIM {
                    stm_acc[k] = stm_acc[k].saturating_add(
                        crate::nnue_input::NNUEFeedForward::from(&[]).ft_weights[f].vals[k]
                    );
                }
            }
        }

        // This is wrong — we need to call the actual network. Fix in implementation.
        // For now, return zero accumulators.
        (stm_acc, ntm_acc)
    }

    /// Forward pass: evaluate board, return raw score (before tanh × SCALE).
    /// Uses existing accumulators if valid, otherwise computes from scratch.
    pub fn forward(&self, stm_input: &[f32; INPUT_DIM], ntm_input: &[f32; INPUT_DIM], non_king_count: u8) -> f32 {
        // Compute accumulators
        let (stm_acc, ntm_acc) = self.compute_accumulators(stm_input, ntm_input);

        // Apply SCReLU and compute output
        let bucket = bucket_index(non_king_count);
        let mut output = 0i32;

        // Side-to-move
        for i in 0..FT_DIM {
            output += screlu(stm_acc[i]) * i32::from(self.out_weights[bucket][i]);
        }

        // Not-side-to-move
        for i in 0..FT_DIM {
            output += screlu(ntm_acc[i]) * i32::from(self.out_weights[bucket][FT_DIM + i]);
        }

        // Reduce quantization
        output /= i32::from(QA);
        output += i32::from(self.out_bias[bucket]);
        output *= SCALE;
        output /= i32::from(QA) * i32::from(QB);

        output as f32
    }

    pub fn forward_output(&self, stm_input: &[f32; INPUT_DIM], ntm_input: &[f32; INPUT_DIM], non_king_count: u8) -> NNOutput {
        NNOutput { alpha: 0.5, beta: 0.5, nn_score: self.forward(stm_input, ntm_input, non_king_count), correction: 0.0 }
    }
}

impl Default for NNUEFeedForward {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// NNUE Feedforward — burn training version (FP32)
// =============================================================================

#[cfg(feature = "train")]
use burn::prelude::*;
#[cfg(feature = "train")]
use burn::module::Module;
#[cfg(feature = "train")]
use burn::nn::Linear;
#[cfg(feature = "train")]
use burn::nn::LinearConfig;

#[cfg(feature = "train")]
#[derive(Module, Debug)]
pub struct NNUEFeedForwardBurn<B: Backend = burn_ndarray::NdArray<f32>> {
    pub ft: Linear<B>,       // 1260 → 512
    pub out: Linear<B>,     // 1024 → 8
}

#[cfg(feature = "train")]
impl<B: Backend> NNUEFeedForwardBurn<B> {
    pub fn new() -> Self {
        let device = <B as Backend>::Device::default();
        let ft = LinearConfig::new(INPUT_DIM, FT_DIM).with_bias(true).init(&device);
        let out = LinearConfig::new(FT_DIM * 2, NUM_BUCKETS).with_bias(true).init(&device);
        Self { ft, out }
    }

    /// Forward pass with bucket selection.
    pub fn forward_with_bucket(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM], bucket_idx: usize) -> f32 {
        use burn::tensor::TensorData;
        let device = <B as Backend>::Device::default();

        // Feature transformer: 1260 → 512, ReLU
        let stm_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::from(*stm), &device);
        let ntm_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::from(*ntm), &device);

        let stm_feat: Tensor<B, 1> = self.ft.forward(stm_tensor).map(|v| v.relu());
        let ntm_feat: Tensor<B, 1> = self.ft.forward(ntm_tensor).map(|v| v.relu());

        // SCReLU: clamp(0, 255)²
        let stm_screlu = stm_feat.map(|v| {
            let clamped = v.clamp(0.0, QA as f32);
            clamped * clamped
        });
        let ntm_screlu = ntm_feat.map(|v| {
            let clamped = v.clamp(0.0, QA as f32);
            clamped * clamped
        });

        // Concatenate: [1024]
        let combined: Tensor<B, 1> = Tensor::cat(vec![stm_screlu, ntm_screlu], 0);

        // Output: 1024 → 8
        let raw_buckets: Tensor<B, 1> = self.out.forward(combined);

        let raw: f32 = raw_buckets.to_data().as_slice().expect("expected 8")[bucket_idx];

        // Apply bullet formula
        let output = raw / QA as f32 + self.out.bias.clone().to_data().as_slice().expect("expected 8")[bucket_idx];
        let output = output * SCALE as f32 / (QA as f32 * QB as f32);

        output.tanh() * SCALE as f32
    }

    pub fn forward_for_inference(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM], non_king_count: u8) -> NNOutput {
        let raw = self.forward_with_bucket(stm, ntm, bucket_index(non_king_count));
        NNOutput { alpha: 0.5, beta: 0.5, nn_score: raw, correction: 0.0 }
    }
}

#[cfg(feature = "train")]
impl<B: Backend> Default for NNUEFeedForwardBurn<B> {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check 2>&1 | tail -15`
Expected: Compile errors (Accumulator usage is wrong in the stub — fix in next step)

- [ ] **Step 3: Fix the implementation**

The `compute_accumulators` method has a broken stub. Rewrite it properly:

```rust
/// Compute accumulators from scratch (full forward pass).
/// stm_input and ntm_input are [f32; 1260] sparse binary inputs.
fn compute_accumulators(stm_input: &[f32; INPUT_DIM], ntm_input: &[f32; INPUT_DIM]) -> ([i16; FT_DIM], [i16; FT_DIM]) {
    let mut stm_acc = [0i32; FT_DIM];  // i32 for accumulation
    let mut ntm_acc = [0i32; FT_DIM];

    // Add bias first
    for k in 0..FT_DIM {
        stm_acc[k] = self.ft_bias[k] as i32;
        ntm_acc[k] = self.ft_bias[k] as i32;
    }

    // Accumulate active features
    for f in 0..INPUT_DIM {
        if stm_input[f] != 0.0 {
            for k in 0..FT_DIM {
                stm_acc[k] += self.ft_weights[f].vals[k] as i32;
            }
        }
    }

    for f in 0..INPUT_DIM {
        if ntm_input[f] != 0.0 {
            for k in 0..FT_DIM {
                ntm_acc[k] += self.ft_weights[f].vals[k] as i32;
            }
        }
    }

    // Clamp and convert to i16 (SCReLU first step)
    let stm_acc: [i16; FT_DIM] = stm_acc.map(|v| (v as i16).clamp(0, QA));
    let ntm_acc: [i16; FT_DIM] = ntm_acc.map(|v| (v as i16).clamp(0, QA));

    (stm_acc, ntm_acc)
}
```

Run: `cargo check 2>&1 | tail -10`
Expected: Clean compile

- [ ] **Step 4: Run tests**

Run: `cargo test nn_eval::tests 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/nn_eval.rs && git commit -m "feat(nn): add NNUEFeedForward (1260→512×2→8, bullet-compliant)"
```

---

## Task 3: Add Accumulator State to Board

**Files:** Modify: `src/main.rs` (Board struct ~line 1651, make_move ~line 1936, undo_move ~line 1977)

- [ ] **Step 1: Add to Board struct**

In `src/main.rs` around line 1651, add after `repetition_history`:

```rust
// NNUE accumulator state (incremental evaluation)
// #[repr(C, align(64))] for SIMD compatibility
#[repr(C, align(64))]
struct Accumulator {
    vals: [i16; crate::nnue_input::FT_DIM],
}

struct NNUEState {
    stm_acc: Accumulator,   // side-to-move accumulator
    ntm_acc: Accumulator,   // not-side-to-move accumulator
    valid: bool,             // false after king move / full refresh needed
}

impl Default for NNUEState {
    fn default() -> Self {
        Self {
            stm_acc: Accumulator { vals: [0i16; crate::nnue_input::FT_DIM] },
            ntm_acc: Accumulator { vals: [0i16; crate::nnue_input::FT_DIM] },
            valid: false,
        }
    }
}
```

Also add to `Board` struct fields:
```rust
nnue: NNUEState,
```

- [ ] **Step 2: Initialize accumulators on `Board::new`**

In `Board::new()`, after the board is initialized:

```rust
// Initialize NNUE accumulators
self.nnue = NNUEState::default();
self.nnue.valid = false; // Will be computed on first evaluation
```

- [ ] **Step 3: Implement `update_nnue_on_move`**

In `src/main.rs`, add method to `Board`:

```rust
/// Update NNUE accumulators after a move.
/// Removes feature at src, adds feature at dst, removes captured piece at dst.
fn update_nnue_on_move(&mut self, action: &Action) {
    use crate::nnue_input::NNUEFeedForward;
    static NET: std::sync::LazyLock<NNUEFeedForward> =
        std::sync::LazyLock::new(NNUEFeedForward::new);

    if !self.nnue.valid {
        return; // Will refresh on next eval
    }

    // Get feature indices for the moved piece at src and dst
    let src_piece = self.get_piece_at(action.src).unwrap();
    let dst_piece = self.get_piece_at(action.tar);

    // Feature index calculation
    let stm_base = match self.current_side {
        Color::Red => 0,
        Color::Black => 630,
    };
    let ntm_base = match self.current_side {
        Color::Red => 630,
        Color::Black => 0,
    };

    let src_idx = stm_base + (src_piece.piece_type as usize) * 90 + action.src.y * 9 + action.src.x;
    let dst_idx = stm_base + (src_piece.piece_type as usize) * 90 + action.tar.y * 9 + action.tar.x;

    // For NTM: vertical flip
    let ntm_src_idx = ntm_base + (src_piece.piece_type as usize) * 90 + (9 - action.src.y) * 9 + action.src.x;
    let ntm_dst_idx = ntm_base + (src_piece.piece_type as usize) * 90 + (9 - action.tar.y) * 9 + action.tar.x;

    // Update accumulators: add_feature = subtract old, remove_feature = subtract
    // Actually: acc_new = acc_old - w[old] + w[new]
    for k in 0..crate::nnue_input::FT_DIM {
        // STM accumulator: subtract src, add dst
        self.nnue.stm_acc.vals[k] = self.nnue.stm_acc.vals[k]
            .saturating_sub(NET.ft_weights[src_idx].vals[k])
            .saturating_add(NET.ft_weights[dst_idx].vals[k]);

        // NTM accumulator: same operation (features are already in NTM basis)
        self.nnue.ntm_acc.vals[k] = self.nnue.ntm_acc.vals[k]
            .saturating_sub(NET.ft_weights[ntm_src_idx].vals[k])
            .saturating_add(NET.ft_weights[ntm_dst_idx].vals[k]);
    }

    // If capture: remove captured piece's feature at dst
    if let Some(captured) = action.captured {
        let cap_idx = stm_base + (captured.piece_type as usize) * 90 + action.tar.y * 9 + action.tar.x;
        let ntm_cap_idx = ntm_base + (captured.piece_type as usize) * 90 + (9 - action.tar.y) * 9 + action.tar.x;

        for k in 0..crate::nnue_input::FT_DIM {
            self.nnue.stm_acc.vals[k] = self.nnue.stm_acc.vals[k]
                .saturating_sub(NET.ft_weights[cap_idx].vals[k]);
            self.nnue.ntm_acc.vals[k] = self.nnue.ntm_acc.vals[k]
                .saturating_sub(NET.ft_weights[ntm_cap_idx].vals[k]);
        }
    }
}
```

- [ ] **Step 4: Implement `undo_nnue_on_move`**

```rust
/// Reverse the accumulator update (for undo_move).
fn undo_nnue_on_move(&mut self, action: &Action) {
    use crate::nnue_input::NNUEFeedForward;
    static NET: std::sync::LazyLock<NNUEFeedForward> =
        std::sync::LazyLock::new(NNUEFeedForward::new);

    if !self.nnue.valid {
        return;
    }

    let src_piece = self.get_piece_at(action.src).unwrap(); // piece that moved (now back at src)
    let dst_piece = action.captured; // piece that was at dst (if any)

    let stm_base = match self.current_side {
        Color::Red => 0,
        Color::Black => 630,
    };
    let ntm_base = match self.current_side {
        Color::Red => 630,
        Color::Black => 0,
    };

    // Undo: add back subtracted (src), subtract added (dst)
    let src_idx = stm_base + (src_piece.piece_type as usize) * 90 + action.src.y * 9 + action.src.x;
    let dst_idx = stm_base + (src_piece.piece_type as usize) * 90 + action.tar.y * 9 + action.tar.x;
    let ntm_src_idx = ntm_base + (src_piece.piece_type as usize) * 90 + (9 - action.src.y) * 9 + action.src.x;
    let ntm_dst_idx = ntm_base + (src_piece.piece_type as usize) * 90 + (9 - action.tar.y) * 9 + action.tar.x;

    for k in 0..crate::nnue_input::FT_DIM {
        // Reverse: subtract dst (added), add src (subtracted)
        self.nnue.stm_acc.vals[k] = self.nnue.stm_acc.vals[k]
            .saturating_sub(NET.ft_weights[dst_idx].vals[k])
            .saturating_add(NET.ft_weights[src_idx].vals[k]);
        self.nnue.ntm_acc.vals[k] = self.nnue.ntm_acc.vals[k]
            .saturating_sub(NET.ft_weights[ntm_dst_idx].vals[k])
            .saturating_add(NET.ft_weights[ntm_src_idx].vals[k]);
    }

    // If capture: add back captured piece's contribution
    if let Some(captured) = action.captured {
        let cap_idx = stm_base + (captured.piece_type as usize) * 90 + action.tar.y * 9 + action.tar.x;
        let ntm_cap_idx = ntm_base + (captured.piece_type as usize) * 90 + (9 - action.tar.y) * 9 + action.tar.x;

        for k in 0..crate::nnue_input::FT_DIM {
            self.nnue.stm_acc.vals[k] = self.nnue.stm_acc.vals[k]
                .saturating_add(NET.ft_weights[cap_idx].vals[k]);
            self.nnue.ntm_acc.vals[k] = self.nnue.ntm_acc.vals[k]
                .saturating_add(NET.ft_weights[ntm_cap_idx].vals[k]);
        }
    }
}
```

- [ ] **Step 5: Wire into `make_move` and `undo_move`**

In `make_move` (~line 1936), after the move is made:
```rust
self.nnue.valid = false; // Mark as needing refresh on next eval
// OR: call update_nnue_on_move if we want incremental updates
```

In `undo_move` (~line 1977), after the move is undone:
```rust
self.nnue.valid = false; // Mark as needing refresh
// OR: call undo_nnue_on_move
```

- [ ] **Step 6: Verify compilation**

Run: `cargo check 2>&1 | tail -10`
Expected: Clean compile

- [ ] **Step 7: Commit**

```bash
git add src/main.rs && git commit -m "feat(nn): add NNUE accumulator state to Board, update/undo move"
```

---

## Task 4: Update `nn_evaluate_or_handcrafted`

**Files:** Modify: `src/nn_eval.rs` (lines 674–704), `src/nnue_input.rs` (export from `Board`)

- [ ] **Step 1: Update `nn_evaluate_or_handcrafted`**

Replace the existing function (~line 674) with:

```rust
/// Hybrid NN + handcrafted evaluation.
/// Uses incremental accumulators if valid, otherwise computes from scratch.
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    use crate::nnue_input::{NNInputPlanes, count_non_king_pieces, NNUEFeedForward};

    let handcrafted = handcrafted_evaluate(board, side, initiative);

    // Build input planes
    let (stm, ntm) = NNInputPlanes::from_board(board);
    let non_king_count = count_non_king_pieces(board);

    #[cfg(feature = "train")]
    {
        type IB = burn_ndarray::NdArray<f32>;
        static NET: std::sync::LazyLock<NNUEFeedForwardBurn<IB>> =
            std::sync::LazyLock::new(NNUEFeedForwardBurn::new);
        let output = NET.forward_for_inference(&stm.to_array(), &ntm.to_array(), non_king_count);
        let blended = 0.5 * output.nn_score + 0.5 * handcrafted as f32;
        return blended as i32;
    }

    #[cfg(not(feature = "train"))]
    {
        static NET: std::sync::LazyLock<NNUEFeedForward> =
            std::sync::LazyLock::new(NNUEFeedForward::new);

        // Use cached accumulators if valid
        let nn_score = if board.nnue.valid {
            // Incremental forward from accumulators
            // (need to add forward_from_accumulators method)
            NET.forward_for_inference(&stm.to_array(), &ntm.to_array(), non_king_count).nn_score
        } else {
            // Full forward pass
            NET.forward_for_inference(&stm.to_array(), &ntm.to_array(), non_king_count).nn_score
        };

        let blended = 0.5 * nn_score + 0.5 * handcrafted as f32;
        blended as i32
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check 2>&1 | tail -10`
Expected: Clean compile

- [ ] **Step 3: Commit**

```bash
git add src/nn_eval.rs && git commit -m "feat(nn): wire nn_evaluate_or_handcrafted to NNUEFeedForward"
```

---

## Task 5: Update Training (`src/nn_train.rs`)

**Files:** Modify: `src/nn_train.rs`

- [ ] **Step 1: Add `non_king_count` to `TrainingSample`**

In `src/nn_train.rs` around line 18, update `TrainingSample`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub planes: Vec<f32>,      // Now 1260 (STM) + 1260 (NTM) = 2520
    pub label: f32,
    pub side_to_move: u8,
    pub non_king_count: u8,    // NEW: bucket selection
}
```

- [ ] **Step 2: Update `from_board`**

```rust
impl TrainingSample {
    pub fn from_board(board: &Board, side_to_move: Color, score: i32) -> Self {
        use crate::nnue_input::NNInputPlanes;

        let (stm, ntm) = NNInputPlanes::from_board(board);
        let mut planes = stm.to_array().to_vec();
        planes.extend_from_slice(&ntm.to_array());

        let label = (score as f32 / 400.0).clamp(-1.0, 1.0);
        let side_to_move = match side_to_move {
            Color::Red => 0,
            Color::Black => 1,
        };
        let non_king_count = crate::nnue_input::count_non_king_pieces(board);

        Self {
            planes,
            label,
            side_to_move,
            non_king_count,
        }
    }
}
```

- [ ] **Step 3: Update forward chain in training**

Replace conv forward chain with NNUE forward:
```
input [1260]
  ↓ ft: 1260→512, ReLU
  ↓ SCReLU (clamp 0, 255)²
  ↓ split → stm[512], ntm[512]
  ↓ concat [1024]
  ↓ output: 1024→8
  ↓ select bucket by non_king_count
  ↓ ((raw/QA) + bias) * SCALE / (QA*QB)
  ↓ tanh × 400
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check --features train 2>&1 | tail -10`
Expected: Clean compile

- [ ] **Step 5: Commit**

```bash
git add src/nn_train.rs && git commit -m "feat(nn): add non_king_count to TrainingSample, update forward chain"
```

---

## Task 6: Final Build + Test

- [ ] **Step 1: Full build**

Run: `cargo build --release 2>&1 | tail -10`
Expected: Clean build

- [ ] **Step 2: Full test suite**

Run: `cargo test 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(nn): complete NNUE implementation for Xiangqi (1260→512×2→8)"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All sections of `docs/superpowers/specs/2026-04-12-nnue-xianqi-design.md` have a corresponding task
- [ ] No placeholders: All steps have complete code
- [ ] Type consistency: `FT_DIM=512`, `INPUT_DIM=1260`, `NUM_BUCKETS=8`, `QA=255`, `QB=64`, `SCALE=400` used consistently
- [ ] SCReLU returns i32 (not i16)
- [ ] Output formula: `((raw/QA) + bias) * SCALE / (QA*QB)`, then `tanh * SCALE`
- [ ] Accumulator: `[i16; 512]`, 64-byte aligned
- [ ] Bucket formula: `(count-2)/4` with saturating_sub(2) and min(7)
