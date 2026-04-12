# Burn Neural Network Training Pipeline — Design Spec

**Status:** Approved 2026-04-10
**Architecture:** Approach B — burn-based CompactResNet for inference + separate nn_train.rs for training
**Training:** Supervised pretraining → self-play fine-tuning, CPU-only
**Design basis:** `docs/superpowers/specs/2026-04-10-nn-eval-design.md`

---

## 1. Overview

Replace the ndarray-based `CompactResNet` in `nn_eval.rs` with a burn `#[derive(Module)]` implementation. Add a training module (`nn_train.rs`) behind a `train` feature flag that supports both self-play data collection and supervised learning from file.

### Design Principles

- **CPU-only:** `burn_ndarray` + `burn_autodiff` backends
- **Single model type:** `CompactResNet<B>` is generic over any burn `Backend`, used for both inference and training
- **Preserve blending formula:** Inference path unchanged — `evaluate()` still calls `nn_evaluate_or_handcrafted()` which calls `CompactResNet::forward()`
- **Progressive training:** Phase 1 (supervised) → Phase 2 (self-play fine-tuning)

---

## 2. Plane Layout

Identical to the current `nn_eval.rs` implementation (documented deviation from the original spec):

| Planes | Content |
|--------|---------|
| 0 | Red King |
| 1-2 | Red Advisors |
| 3-4 | Red Elephants |
| 5-6 | Red Horses |
| 7-8 | Red Cannons |
| 9-13 | Red Pawns (by file 0-4) |
| 14 | Black King |
| 15-16 | Black Advisors |
| 17-18 | Black Elephants |
| 19-20 | Black Horses |
| 21-22 | Black Cannons |
| 23-27 | Black Pawns (by file 0-4) |
| 28 | Side-to-move (Red=1, Black=0) |
| 29 | Game phase (0.0→1.0) |
| 30 | Repetition count (0.0→1.0, 3+ reps = 1.0) |
| 31 | Rule-60 counter (0→90+ normalized to 0→1) |
| 32-33 | Reserved zeros |

**Memory layout:** `data[plane * 90 + y * 9 + x]` — plane-first, same as current `InputPlanes`.

---

## 3. Network Architecture

### 3.1 Compact ResNet (burn Module)

```
InputPlanes [1, 38, 9, 10]
    ↓
Initial Conv2d: 38→32 channels, 3×3 kernel, padding=1, bias
    ↓ ReLU
ResBlock 1: 32→32 ch, identity skip
ResBlock 2: 32→32 ch, identity skip
ResBlock 3: 32→64 ch, 1×1 conv skip (downsample)
ResBlock 4: 64→64 ch, identity skip
ResBlock 5: 64→64 ch, identity skip
ResBlock 6: 64→64 ch, identity skip
    ↓
GlobalAvgPool → [1, 64]
Dense: 64→64, ReLU
    ↓
4 output heads (Linear 64→1):
  alpha:     sigmoid → ×0.9 + 0.05 → [0.05, 0.95]
  beta:      sigmoid → ×0.9 + 0.05 → [0.05, 0.95]
  nn_score:  tanh → ×400 → [-400, 400] centipawns
  correction: tanh → ×400 → [-400, 400] centipawns
```

### 3.2 Residual Block

```rust
#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
    // None for identity skip, Some for 1×1 channel change
    skip: Option<nn::conv::Conv2d<B>>,
}
```

Forward: `h = relu(conv1(x)); h = conv2(h); h += skip(x); relu(h)`

### 3.3 Output Heads

Each head is a separate `nn::Linear<B>` with 1 output. Output scaling applied after the raw linear pass:
- **Alpha/Beta:** `sigmoid(linear_out) * 0.9 + 0.05` → range [0.05, 0.95]
- **NN score:** `tanh(linear_out) * 400.0` → range [-400, 400]
- **Correction:** `tanh(linear_out) * 400.0` → range [-400, 400]

---

## 4. Training Pipeline

### 4.1 Phase 1 — Supervised Pretraining

**Data generation:**
1. Run engine search (depth 4-6) on random starting positions
2. Collect ~500K positions with (board_state, search_score) pairs
3. Normalize: `label = clamp(score / 400.0, -1.0, 1.0)`

**Training:**
- Loss: MSE on `nn_score` head only
- Alpha/beta heads: **fixed** (no gradient, initialized to output ≈ 0.5)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Epochs: 15, batch_size: 256

### 4.2 Phase 2 — Self-Play Fine-Tuning

**Data generation:**
1. Self-play games: collect (position, game_outcome) pairs
2. Outcome: +1.0 (win), 0.0 (draw), -1.0 (loss)

**Training:**
- Loss: `mse(nn_score, outcome) + 0.1 * (cross_entropy(alpha, outcome) + cross_entropy(beta, outcome))`
- All heads trainable
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4

---

## 5. Module Structure

### Files

**`src/nn_eval.rs`** (modified)
- Replaces ndarray-based `CompactResNet` with burn `#[derive(Module)]` version
- `CompactResNet<B: Backend>` is generic — inference uses `NdArrayBackend`
- `InputPlanes::from_board()` unchanged
- `nn_evaluate_or_handcrafted()` unchanged from current implementation
- `CompactResNet::forward_for_inference()` takes flat `[f32; 3420]`, returns `NNOutput`

**`src/nn_train.rs`** (new, `#[cfg(feature = "train")]`)
- `TrainingSample` struct with `planes`, `label`, `side_to_move`
- `SelfPlayCollector` for in-memory game data collection
- `train_supervised()` — Phase 1 training
- `train_selfplay()` — Phase 2 training
- `load_training_file()` — binary file loader via bincode
- `export_training_data()` — write positions to binary file
- CLI training menu in existing interactive prompt

### Feature Flag

```toml
[features]
train = ["burn/burn-ndarray", "burn/burn-autodiff", "burn/nn", "burn/optim"]
```

- `cargo build` — inference-only, no burn deps in critical path
- `cargo build --features train` — includes training module

---

## 6. Blending Formula (unchanged)

```
final_score = (alpha / (alpha + beta)) * nn_score
            + (beta / (alpha + beta)) * handcrafted_score
            + correction
```

- alpha, beta ∈ [0.05, 0.95] — sigmoid-clamped, minimum 0.05 floor
- nn_score, correction ∈ [-400, 400] centipawns
- handcrafted_score cast to f32

---

## 7. CLI Training Menu

Added to the existing interactive menu (before game starts):

```
=== 中国象棋引擎 ===
1. ...
2. ...
3. ...
T. 训练模式
```

Training sub-menu:
```
【训练模式】
1. 自我对弈收集 + 训练 (collect then train Phase 1)
2. 从文件加载数据训练 (supervised Phase 1)
3. 继续训练 (load weights + self-play Phase 2)
4. 导出棋谱为训练数据 (export positions to file)
5. 返回主菜单
```

---

## 8. Data Format

Binary file via bincode:

```rust
struct TrainingSample {
    planes: [f32; 3420],  // flat InputPlanes data
    label: f32,            // game outcome +1/0/-1 or clamp(score/400, -1, 1)
    side_to_move: u8,      // 0=Red, 1=Black
}
```

Header: `[u32 count]` followed by `count` serialized `TrainingSample` structs.

---

## 9. Key Implementation Notes

### burn Backend Choice

```rust
use burn::backend::{NdArray, Autodiff};
type TrainingBackend = Autodiff<NdArray>;
type InferenceBackend = NdArray;
```

For inference, `Autodiff` is unnecessary overhead — use plain `NdArray`. For training, use `Autodiff<NdArray>`.

### Weight Serialization

burn's `Module` derive generates a `Record` type. Use burn's built-in serialization:
```rust
// Save
net.save_file(path, &mut File::create(path)?)?;
// Load
let net = CompactResNet::<NdArray>::load_file(path)?;
```

### InputPlanes → burn Tensor

```rust
fn forward_for_inference(&self, planes: &InputPlanes) -> NNOutput {
    let tensor = Tensor::<NdArray, 4>::from_data(
        TensorData::from(planes.data).reshape([1, 38, 9, 10])
    );
    // ... rest of forward pass
}
```

### Alpha/Beta Head Clamping

```rust
let alpha = (alpha_raw.sigmoid() * 0.9 + 0.05).max(0.05);
let beta = (beta_raw.sigmoid() * 0.9 + 0.05).max(0.05);
```

Minimum clamp at 0.05 ensures neither component is ever completely ignored.

---

## 10. Testing

1. **Inference sanity:** `CompactResNet::forward()` at starting position produces near-zero score
2. **Output range:** alpha, beta ∈ [0.05, 0.95]; nn_score, correction ∈ [-400, 400]
3. **Forward/backward:** training loop completes one epoch without NaN
4. **Weight save/load:** saved weights produce identical inference output
5. **Self-play collection:** collector produces legal moves and valid game outcomes
