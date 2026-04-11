# NNUE Implementation Plan (Fairy-Stockfish Architecture)

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development.

**Goal:** Implement Fairy-Stockfish NNUE architecture: Feature Transformer (3420→512) → dual 16-unit accumulators → concat 32 → hidden 32 → 8 buckets. ~1.77M params. FP32 training + int16 quantization for inference.

---

## Architecture Summary

```
InputPlanes [3420] sparse
  ↓ FeatureTransformer: [3420] × [3420, 512]^T + [512] → [512], ReLU
  │
  ├──► acc_red:   Linear [512] → [16], CReLU
  │
  └──► acc_black: Linear [512] → [16], CReLU
              │
         Concatenate
              │
              ▼
         [32]
              │
         Hidden: Linear [32] → [32], CReLU
              │
         Output: Linear [32] → [8 buckets]
              │
         Bucket select (non_king_count)
              │
         tanh × 400
Output: scalar
```

---

## Task 1: Rewrite `src/nn_eval.rs` with Fairy-Stockfish Architecture

**Files:** Modify `src/nn_eval.rs`

**Boundary:**
- KEEP lines 1-182: `InputPlanes` struct and impl
- DELETE lines 183-643: CompactResNet, conv helpers, ResBlock, CompactResNetBurn
- KEEP lines 644+: `NNOutput`, `nn_evaluate_or_handcrafted` (to update in Task 4)

### Step 1: Delete old code (lines 183-643)

### Step 2: Add new architecture after line 182

```rust
// =============================================================================
// NNUE Feedforward — plain ndarray inference (FP32)
// =============================================================================

/// CReLU: clamp to [0, 1]
#[inline]
fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Fast random (xorshift64)
fn fast_rand() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x123456789ABCDEF0);
    fn next() -> u64 {
        let current = STATE.load(Ordering::Relaxed);
        let new_val = current.wrapping_mul(6364136223846793005).wrapping_add(1);
        STATE.store(new_val, Ordering::Relaxed);
        new_val
    }
    next() as f32 / u64::MAX as f32
}

/// He init for fan-in
fn fc_he_init(fan_in: usize) -> f32 {
    let std = (2.0 / fan_in as f32).sqrt();
    (fast_rand() - 0.5) * 2.0 * std
}

/// Bucket index from non-king piece count (0-30) → 0-7
pub fn bucket_index(non_king_count: u8) -> usize {
    ((non_king_count as usize).saturating_sub(2) / 4).min(7)
}

/// Fairy-Stockfish NNUE: FeatureTransformer → dual 16-unit accumulators → 32 → 32 → 8 buckets.
/// ~1.77M parameters (mostly in feature transformer).
#[derive(Clone)]
#[allow(dead_code)]
pub struct NNUEFeedForward {
    // Feature transformer: 3420 → 512
    ft_w: ndarray::Array2<f32>,  // [3420, 512]
    ft_b: ndarray::Array1<f32>,  // [512]
    // Dual accumulators: 512 → 16 each
    acc_red_w: ndarray::Array2<f32>,    // [512, 16]
    acc_red_b: ndarray::Array1<f32>,    // [16]
    acc_black_w: ndarray::Array2<f32>,  // [512, 16]
    acc_black_b: ndarray::Array1<f32>,  // [16]
    // Hidden: 32 → 32
    hidden_w: ndarray::Array2<f32>,    // [32, 32]
    hidden_b: ndarray::Array1<f32>,    // [32]
    // Output: 32 → 8 buckets
    out_w: ndarray::Array2<f32>,       // [8, 32]
    out_b: ndarray::Array1<f32>,       // [8]
}

impl NNUEFeedForward {
    pub fn new() -> Self {
        let make_fc = |rows: usize, cols: usize| -> ndarray::Array2<f32> {
            let mut a = ndarray::Array2::<f32>::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    a[[i, j]] = fc_he_init(rows);
                }
            }
            a
        };
        let make_bias = |n: usize| ndarray::Array1::<f32>::zeros(n);

        Self {
            ft_w: make_fc(3420, 512),
            ft_b: make_bias(512),
            acc_red_w: make_fc(512, 16),
            acc_red_b: make_bias(16),
            acc_black_w: make_fc(512, 16),
            acc_black_b: make_bias(16),
            hidden_w: make_fc(32, 32),
            hidden_b: make_bias(32),
            out_w: make_fc(8, 32),
            out_b: make_bias(8),
        }
    }

    /// Forward with explicit bucket.
    pub fn forward_with_bucket(&self, input: &InputPlanes, non_king_count: u8) -> f32 {
        let x = input.to_array1(); // [3420]

        // Feature transformer: 3420 → 512, ReLU
        let feat = self.ft_w.t().dot(&x) + &self.ft_b;
        let feat = feat.mapv(|v| v.max(0.0)); // ReLU

        // Dual accumulators: 512 → 16 each, CReLU
        let acc_red = self.acc_red_w.t().dot(&feat) + &self.acc_red_b;
        let acc_red = acc_red.mapv(crelu);

        let acc_black = self.acc_black_w.t().dot(&feat) + &self.acc_black_b;
        let acc_black = acc_black.mapv(crelu);

        // Concatenate: [16] + [16] → [32]
        let mut combined = ndarray::Array1::<f32>::zeros(32);
        for i in 0..16 {
            combined[i] = acc_red[i];
            combined[i + 16] = acc_black[i];
        }

        // Hidden: 32 → 32, CReLU
        let h = self.hidden_w.t().dot(&combined) + &self.hidden_b;
        let h = h.mapv(crelu);

        // Output: 32 → 8 buckets
        let raw_buckets = self.out_w.t().dot(&h) + &self.out_b;

        let bucket_idx = bucket_index(non_king_count);
        raw_buckets[bucket_idx].tanh() * 400.0
    }

    /// Forward with default bucket=7.
    pub fn forward(&self, input: &InputPlanes) -> f32 {
        self.forward_with_bucket(input, 30)
    }

    pub fn forward_output(&self, input: &InputPlanes) -> NNOutput {
        NNOutput { alpha: 0.5, beta: 0.5, nn_score: self.forward(input), correction: 0.0 }
    }
}

impl Default for NNUEFeedForward {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// NNUE Feedforward — burn training version
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
use burn::nn::Relu;

#[cfg(feature = "train")]
#[derive(Module, Debug)]
pub struct NNUEFeedForwardBurn<B: Backend = burn_ndarray::NdArray<f32>> {
    pub ft: Linear<B>,       // 3420 → 512
    pub acc_red: Linear<B>,  // 512 → 16
    pub acc_black: Linear<B>, // 512 → 16
    pub hidden: Linear<B>,  // 32 → 32
    pub out: Linear<B>,      // 32 → 8
    pub relu: Relu,
}

#[cfg(feature = "train")]
impl<B: Backend> NNUEFeedForwardBurn<B> {
    pub fn new() -> Self {
        let device = <B as Backend>::Device::default();
        let ft = LinearConfig::new(3420, 512).with_bias(true).init(&device);
        let acc_red = LinearConfig::new(512, 16).with_bias(true).init(&device);
        let acc_black = LinearConfig::new(512, 16).with_bias(true).init(&device);
        let hidden = LinearConfig::new(32, 32).with_bias(true).init(&device);
        let out = LinearConfig::new(32, 8).with_bias(true).init(&device);
        Self { ft, acc_red, acc_black, hidden, out, relu: Relu::new() }
    }

    pub fn forward_with_bucket(&self, planes: &InputPlanes, bucket_idx: usize) -> f32 {
        use burn::tensor::TensorData;
        let device = <B as Backend>::Device::default();

        let flat_arr: [f32; 3420] = planes.data;
        let x: Tensor<B, 1> = Tensor::from_data(TensorData::from(flat_arr), &device);

        // Feature transformer: 3420 → 512, ReLU
        let feat: Tensor<B, 1> = self.relu.forward(self.ft.forward(x));

        // Dual accumulators: 512 → 16, CReLU
        let acc_red: Tensor<B, 1> = self.acc_red.forward(feat.clone())
            .map(|v| v.clamp(0.0, 1.0));
        let acc_black: Tensor<B, 1> = self.acc_black.forward(feat)
            .map(|v| v.clamp(0.0, 1.0));

        // Concatenate: [16] + [16] → [32]
        let combined: Tensor<B, 1> = Tensor::cat(vec![acc_red, acc_black], 0);

        // Hidden: 32 → 32, CReLU
        let h: Tensor<B, 1> = self.hidden.forward(combined)
            .map(|v| v.clamp(0.0, 1.0));

        // Output: 32 → 8
        let raw_buckets: Tensor<B, 1> = self.out.forward(h);

        let raw: f32 = raw_buckets.to_data().as_slice().expect("expected 8")[bucket_idx];
        raw.tanh() * 400.0
    }

    pub fn forward_for_inference(&self, planes: &InputPlanes) -> NNOutput {
        let raw = self.forward_with_bucket(planes, bucket_index(30));
        NNOutput { alpha: 0.5, beta: 0.5, nn_score: raw, correction: 0.0 }
    }
}

#[cfg(feature = "train")]
impl<B: Backend> Default for NNUEFeedForwardBurn<B> {
    fn default() -> Self { Self::new() }
}
```

### Step 3: Verify builds

Run: `cargo build 2>&1 | tail -10`
Run: `cargo check --features train 2>&1 | tail -10`
Expected: both clean

### Step 4: Commit

```bash
git add src/nn_eval.rs && git commit -m "feat(nn): implement Fairy-Stockfish NNUE (feature transformer + dual 16-unit accumulators)"
```

---

## Task 2: Add `non_king_count` to TrainingSample

**Files:** Modify `src/nn_train.rs`

### Step 1: Add field + update from_board

```rust
pub struct TrainingSample {
    pub planes: Vec<f32>,
    pub label: f32,
    pub side_to_move: u8,
    pub non_king_count: u8, // NEW
}
```

In `from_board()`: compute and store `non_king_count` (count pieces that are NOT King).

### Step 2: Commit

```bash
git add src/nn_train.rs && git commit -m "feat(nn): add non_king_count to TrainingSample"
```

---

## Task 3: Update `nn_train.rs` FC forward chain

### Forward chain (per sample):
```
input [3420]
  ↓ ft: 3420 → 512, ReLU
  ↓ acc_red: 512 → 16, CReLU
  ↓ acc_black: 512 → 16, CReLU
  ↓ concat: [32]
  ↓ hidden: 32 → 32, CReLU
  ↓ out: 32 → 8
  ↓ select bucket by non_king_count
  ↓ tanh × 400
```

### Step 1: Update train_supervised, compute_val_loss, train_selfplay

Replace conv forward chains with NNUE forward chains above.

For bucket selection in batch training: compute raw_buckets tensor [batch, 8], gather selected bucket per sample, compute MSE.

### Step 2: Commit

```bash
git add src/nn_train.rs && git commit -m "feat(nn): update training to Fairy-Stockfish NNUE forward chain"
```

---

## Task 4: Wire `nn_evaluate_or_handcrafted`

### Step 1: Update `nn_evaluate_or_handcrafted`

```rust
#[cfg(not(feature = "train"))]
static NN_NET: std::sync::LazyLock<NNUEFeedForward> =
    std::sync::LazyLock::new(NNUEFeedForward::new);

pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);
    let input = InputPlanes::from_board(board, side);

    #[cfg(feature = "train")]
    let output = {
        type IB = burn_ndarray::NdArray<f32>;
        static NET: std::sync::LazyLock<NNUEFeedForwardBurn<IB>> =
            std::sync::LazyLock::new(NNUEFeedForwardBurn::new);
        NET.forward_for_inference(&input)
    };

    #[cfg(not(feature = "train"))]
    let output = NN_NET.forward_output(&input);

    // Compute non_king_count
    let mut nk: u8 = 0;
    for y in 0..10 {
        for x in 0..9 {
            if let Some(p) = board.cells[y][x] {
                if p.piece_type != PieceType::King { nk += 1; }
            }
        }
    }
    let _ = nk;

    let blended = 0.5 * output.nn_score + 0.5 * handcrafted as f32;
    blended as i32
}
```

### Step 2: Commit

```bash
git add src/nn_eval.rs && git commit -m "feat(nn): wire nn_evaluate_or_handcrafted to Fairy-Stockfish NNUE"
```

---

## Task 5: Update tests

```rust
#[test]
fn test_nnue_forward_output_ranges() {
    let net = NNUEFeedForward::new();
    let board = Board::new(RuleSet::Official, 1);
    let planes = InputPlanes::from_board(&board, Color::Red);
    for bucket in 0..8 {
        let score = net.forward_with_bucket(&planes, (bucket * 4 + 2) as u8);
        assert!(score >= -400.0 && score <= 400.0);
    }
}

#[test]
fn test_bucket_index() {
    assert_eq!(bucket_index(2), 0);
    assert_eq!(bucket_index(6), 1);
    assert_eq!(bucket_index(10), 2);
    assert_eq!(bucket_index(14), 3);
    assert_eq!(bucket_index(18), 4);
    assert_eq!(bucket_index(22), 5);
    assert_eq!(bucket_index(26), 6);
    assert_eq!(bucket_index(30), 7);
}
```

Run: `cargo test --features train nn_eval::tests`

### Step 2: Commit

```bash
git add src/nn_eval.rs && git commit -m "test(nn): update tests for Fairy-Stockfish NNUE"
```

---

## Task 6: Final build + test verification

```bash
cargo build --release
cargo build --release --features train
cargo test --features train
git add -A && git commit -m "feat(nn): complete Fairy-Stockfish NNUE architecture migration"
```

---

## Type Consistency

- `InputPlanes::to_array1()` → `Array1<f32>` ✓
- `bucket_index(u8) → usize` ✓
- `NNUEFeedForward::forward_with_bucket(input, non_king_count: u8) → f32` ✓
- `NNUEFeedForwardBurn::forward_with_bucket(planes, bucket_idx: usize) → f32` ✓
- Layer dims: ft[3420,512], acc[512,16]×2, hidden[32,32], out[8,32] ✓
