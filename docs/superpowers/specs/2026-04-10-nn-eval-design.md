# Neural Network Evaluation Module — Design Spec

**Status:** Approved 2026-04-10
**Architecture:** Hybrid NN + handcrafted blend with learned alpha blending
**Training:** Self-play + supervised bootstrap, CPU-only

---

## 1. Overview

Replace the engine's static handcrafted evaluation with a **hybrid neural network + handcrafted evaluator**. The NN learns positional corrections from self-play and supervised data, while the handcrafted eval provides stable baseline evaluation. A learned alpha-beta blending mechanism determines how much to trust each component per-position.

### Design Principles

- **CPU-optimized:** Compact ResNet (~180K params) designed for fast inference on CPU
- **Learned blending:** NN outputs per-position alpha/beta weights — not fixed blending
- **Preserve existing eval:** All handcrafted components remain; NN learns corrections
- **Progressive training:** Supervised pretraining → self-play fine-tuning

### Pipeline

```
Board → InputPlanes → CompactResNet → (alpha, beta, correction)
                                              ↓
handcrafted_evaluate(board, side) ─────────────→ blended_score
```

**Blending formula:**
```
final_score = (alpha / (alpha + beta)) * nn_score
            + (beta / (alpha + beta)) * handcrafted_score
            + correction * scale
```

Where:
- `alpha`, `beta` ∈ [0, 1] — learned sigmoid weights (never completely zero)
- `correction` ∈ [-1, 1] — additive centipawn offset, scaled by 400 cp
- `nn_score` = network's raw prediction scaled to centipawns

---

## 2. Input Representation

### 2.1 Piece Planes (AlphaZero Style)

**Red piece planes (16):**
| Plane | Content |
|-------|---------|
| 0 | King (1 where Red King exists) |
| 1-2 | Advisors (1 where Red Advisors exist) |
| 3-4 | Elephants (1 where Red Elephants exist) |
| 5-6 | Horses (1 where Red Horses exist) |
| 7-8 | Cannons (1 where Red Cannons exist) |
| 9-13 | Pawns (1 where Red Pawns exist, per file) |

**Black piece planes (16):** Mirrored encoding, same structure.

**Side-to-move plane (1):** All 1s if Red to move, all 0s if Black to move.

**Total piece planes: 33**

### 2.2 Auxiliary Feature Planes (5)

| Plane | Description | Range |
|-------|-------------|-------|
| 33 | Game phase | 0.0 (opening) → 1.0 (endgame) |
| 34 | Repetition count | 0.0 (0 reps) → 1.0 (3+ reps) |
| 35 | Rule-60 counter | 0.0 (0) → 1.0 (90+) |
| 36 | King safety diff | [-1, 1] (our - their) / 100 |
| 37 | Material imbalance | [-1, 1] (our - their) / 1000 |

### 2.3 Implementation

```rust
/// Input to the neural network: flat array of 38 planes at 9×10
#[derive(Clone)]
pub struct InputPlanes {
    pub data: [f32; 38 * 9 * 10],  // 3420 values, ~14KB
}

impl InputPlanes {
    /// Convert a Board to input planes
    pub fn from_board(board: &Board, side_to_move: Color) -> Self {
        let mut data = [0.0f32; 38 * 9 * 10];
        
        // Piece planes (0-31): populate from board.cells
        // ... (see implementation)
        
        // Plane 32: side to move
        let stm_offset = 32 * 9 * 10;
        if side_to_move == Color::Red {
            for i in 0..(9 * 10) { data[stm_offset + i] = 1.0; }
        }
        
        // Auxiliary planes (33-37)
        // ... (see implementation)
        
        Self { data }
    }
}
```

**Memory layout:** `data[plane * 90 + y * 9 + x]` — plane-first for efficient CNN convolution.

---

## 3. Network Architecture

### 3.1 Compact ResNet (8 Layers, 128 Channels)

```
InputPlanes (38 × 9 × 10)
    ↓
Initial Conv: 38 → 64 channels, 3×3 kernel, padding 1
    ↓ BatchNorm → ReLU
    ↓
Residual Block 1: 64 ch, 3×3 kernel + skip (identity)
    ↓
Residual Block 2: 64 ch, 3×3 kernel + skip (identity)
    ↓ (downsample: 64 → 128 via 1×1 conv on skip)
Residual Block 3: 128 ch, 3×3 kernel + skip (1×1 conv)
    ↓
Residual Block 4: 128 ch, 3×3 kernel + skip (identity)
    ↓
Residual Block 5: 128 ch, 3×3 kernel + skip (identity)
    ↓
Residual Block 6: 128 ch, 3×3 kernel + skip (identity)
    ↓
Residual Block 7: 128 ch, 3×3 kernel + skip (identity)
    ↓
Residual Block 8: 128 ch, 3×3 kernel + skip (identity)
    ↓
Global Average Pooling (9×10 → 1×1) → 128-dim vector
    ↓
Dense: 128 → 64 → ReLU
    ↓
Output Heads:
    ├── Alpha Head:  64 → 1 → Sigmoid → range [0.05, 0.95]
    ├── Beta Head:  64 → 1 → Sigmoid → range [0.05, 0.95]
    └── Score Head: 64 → 1 → Tanh → range [-1, 1]
```

**Alpha/Beta minimum clamp:** 0.05 — ensures neither component is ever completely ignored.

### 3.2 Residual Block Detail

```rust
struct ResidualBlock {
    conv1: Conv2D(64→64, 3×3),
    bn1: BatchNorm,
    conv2: Conv2D(64→64, 3×3),
    bn2: BatchNorm,
}

impl Forward for ResidualBlock {
    fn forward(&self, x: &Array4<f32>) -> Array4<f32> {
        let residual = x;
        let x = self.bn1.forward(&relu(&self.conv1.forward(x)));
        let x = self.bn2.forward(&self.conv2.forward(x));
        x + residual  // Skip connection
    }
}
```

### 3.3 Output Scaling

- **Alpha/Beta:** `sigmoid(output) * 0.9 + 0.05` → range [0.05, 0.95]
- **Score:** `tanh(output) * 400.0` → range [-400, +400] centipawns

### 3.4 Parameter Count

| Layer | Parameters |
|-------|------------|
| Initial conv | 38×64×3×3 + 64 = ~22K |
| Blocks 1-2 (64 ch) | 2 × (64×64×3×3×2 + 64×2) ≈ 148K |
| Blocks 3-8 (128 ch) | 6 × (128×128×3×3×2 + 128×2) ≈ 1.4M |
| Dense + heads | 128×64 + 64 + 64×1×3 + 3 ≈ 8K |
| **Total** | **~1.6M parameters** |

Wait — with 128 channels the parameter count is ~1.6M, not 180K. For CPU inference speed, we target ~180K params instead:

**Revised Architecture (128K params):**
- Initial conv: 38 → 32 channels
- Blocks 1-2: 32 channels (~25K params)
- Blocks 3-4: 32 → 64 channels (~100K params)
- Blocks 5-6: 64 channels (~50K params)
- Global pool → 64 → 32 → 3 heads (~3K params)
- **Total: ~180K params**

---

## 4. Training Pipeline

### Phase 1: Supervised Pretraining (CPU)

**Data generation:**
1. Run engine with current search (depth 4-6) on random starting positions
2. Collect ~500K positions with (board_state, search_score) pairs
3. Normalize scores: `label = clamp(search_score / 400.0, -1.0, 1.0)`

**Training:**
```rust
let batch_size = 256;
let epochs = 15;
let lr = 1e-3;
let weight_decay = 1e-4;
let optimizer = AdamW::new(lr, weight_decay);

// Loss: MSE on score head only
// Alpha/beta heads initialized to 0.5, fixed during pretrain
fn loss(pred: &NNOutput, label: f32) -> f32 {
    let target_score = label;  // [-1, 1]
    (pred.score - target_score).powi(2)
}
```

**Training time estimate (CPU):** ~2-3 days for 500K positions × 15 epochs.

### Phase 2: Self-Play Fine-Tuning (CPU — SLOW)

**Data generation:**
1. Start from pretrain weights
2. Self-play games: ~10K games total (CPU-limited)
3. Collect (position, game_outcome) pairs: +1 (win), 0 (draw), -1 (loss)

**Training:**
```rust
// Loss: MSE on score head + cross-entropy on outcome
fn loss(pred: &NNOutput, outcome: f32) -> f32 {
    let score_loss = (pred.score - outcome).powi(2);
    let alpha_ce = -outcome * pred.alpha.ln() - (1.0 - outcome) * (1.0 - pred.alpha).ln();
    let beta_ce = -outcome * pred.beta.ln() - (1.0 - outcome) * (1.0 - pred.beta).ln();
    score_loss + 0.1 * (alpha_ce + beta_ce)
}
```

**Training time estimate (CPU):** ~2-4 weeks for 10K games. This is the bottleneck.

### Phase 3: Online Tuning (Optional)

- Run engine vs. previous version
- Collect positions where new version outperforms old
- Incremental retraining

---

## 5. Integration with Search

### 5.1 Interleaved Evaluation

```rust
// In search: evaluate every Nth node with NN, rest with handcrafted
const NN_EVAL_INTERVAL: usize = 5;  // Tunable

fn search_node(board: &mut Board, depth: u8, mut alpha: i32, beta: i32) -> i32 {
    static mut NN_NODE_COUNTER: usize = 0;
    
    unsafe {
        NN_NODE_COUNTER += 1;
        let use_nn = NN_NODE_COUNTER % NN_EVAL_INTERVAL == 0 && NN_ENABLED;
        
        let eval = if use_nn {
            nn_evaluate(board, board.current_side)
        } else {
            handcrafted_evaluate(board, board.current_side, false)
        };
        // ...
    }
}
```

### 5.2 evaluate() Wrapper

```rust
pub fn evaluate(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);
    
    if !NN_ENABLED {
        return handcrafted;
    }
    
    let input = InputPlanes::from_board(board, side);
    let output = NN_FORWARD(&input);
    
    // Normalize alpha + beta to sum to 1 (with minimum floor)
    let alpha = output.alpha.max(0.05);
    let beta = output.beta.max(0.05);
    let total = alpha + beta;
    let alpha_norm = alpha / total;
    let beta_norm = beta / total;
    
    // NN outputs score in centipawns already scaled
    let nn_score = output.score;
    let correction = output.correction * 400.0;  // Scale correction
    
    let blended = alpha_norm * nn_score + beta_norm * handcrafted as f32 + correction;
    blended as i32
}
```

### 5.3 Configuration

```rust
// In config or main.rs
const NN_ENABLED: bool = true;
const NN_EVAL_INTERVAL: usize = 5;  // 1 = every node (slow), 10 = every 10th (fast)
const NN_WEIGHTS_PATH: &str = "data/nn_weights.bin";
```

---

## 6. Module Structure

### New Files

**`src/nn_eval.rs`** (~400 lines)
```rust
pub mod nn_eval {
    // InputPlanes: board → flat array conversion
    // Conv2D, BatchNorm, ReLU, ResidualBlock: layer types
    // CompactResNet: full network with forward pass
    // nn_evaluate(board, side) -> NNOutput
    // Weight serialization: save/load to binary
}
```

**`src/eval.rs`** (~600 lines) — extracted from main.rs
```rust
pub mod eval {
    // All existing handcrafted eval components
    pub fn handcrafted_evaluate(board: &Board, side: Color, initiative: bool) -> i32;
    // (Extracted unchanged from current main.rs eval module)
}
```

### Modified Files

**`src/main.rs`**
- Replace `pub mod eval { ... }` with:
  ```rust
  mod eval;
  mod nn_eval;
  ```
- Update `evaluate()` to dispatch to NN or handcrafted
- Add NN config constants and startup weight loading

**`Cargo.toml`** — add dependencies:
```toml
ndarray = "0.16"      # N-dimensional arrays for NN
serde = "1.0"         # Weight serialization
```

### File Map

| File | Lines | Purpose |
|------|-------|---------|
| `src/nn_eval.rs` | ~400 NEW | NN evaluation module |
| `src/eval.rs` | ~600 NEW (extracted) | Handcrafted eval extracted |
| `src/main.rs` | -50 changed | Module references, dispatch |
| `Cargo.toml` | +4 changed | Added deps |
| `data/nn_weights.bin` | — | Trained weights (gitignore) |

---

## 7. Files Affected Summary

- **NEW:** `src/nn_eval.rs` — Neural network evaluation
- **NEW:** `src/eval.rs` — Extracted handcrafted evaluation
- **MODIFIED:** `src/main.rs` — Module integration, evaluate() dispatch
- **MODIFIED:** `Cargo.toml` — Added `ndarray`, `serde` dependencies

All existing evaluation logic preserved; NN learns corrections on top.

---

## 8. Testing

1. **Unit test: InputPlanes roundtrip** — board → planes → board should be lossless for piece info
2. **Integration test: NN loads weights** — verify weights deserialize correctly
3. **Sanity test: NN eval at start** — NN eval at starting position should be near 0.0 (equal)
4. **Regression test: handcrafted unchanged** — compare `handcrafted_evaluate()` output before/after refactor
5. **Search test: interleaved eval** — verify search produces legal moves and doesn't crash
6. **Self-play test: NN eval vs handcrafted** — run 100 self-play games with NN enabled, verify they complete

---

## 9. Open Questions

- **Phase 2 self-play games:** Is 10K feasible on CPU, or should we target fewer (1-2K)?
- **NN_EVAL_INTERVAL:** Start with 5 — tune based on speed measurement
- **Training infrastructure:** Is there existing infra for running CPU-intensive training jobs, or ad-hoc?

---

## Appendix A: Full ResNet Forward Pass (Pseudocode)

```rust
impl CompactResNet {
    fn forward(&self, input: &InputPlanes) -> NNOutput {
        // input: [38, 9, 10] → reshape to [1, 38, 9, 10]
        let mut x = input.to_array4();
        
        // Initial conv: 38 → 64
        x = self.initial_conv.run(&x);
        
        // Residual blocks
        x = self.block1.forward(&x);  // 64, skip=identity
        x = self.block2.forward(&x);  // 64, skip=identity
        x = self.block3.forward(&x);  // 64→128, skip=1×1 conv
        x = self.block4.forward(&x);  // 128, skip=identity
        x = self.block5.forward(&x);  // 128, skip=identity
        x = self.block6.forward(&x);  // 128, skip=identity
        
        // Global average pool: [1, 128, 9, 10] → [1, 128, 1, 1] → [1, 128]
        let x = x.mean_axis(Axis(2)).mean_axis(Axis(2));
        
        // Dense: 128 → 64
        let x = self.dense1.run(&x);
        
        // Heads
        let alpha = sigmoid(self.alpha_head.run(&x)) * 0.9 + 0.05;
        let beta = sigmoid(self.beta_head.run(&x)) * 0.9 + 0.05;
        let score = tanh(self.score_head.run(&x));
        
        NNOutput { alpha, beta, score }
    }
}
```
