# NNUE-Style Neural Network — Design Spec (Fairy-Stockfish Architecture)

## Status
Revised to match Fairy-Stockfish's proven NNUE architecture.

---

## 1. Overview

Replace the convolutional ResNet with Fairy-Stockfish's proven NNUE architecture:
- **Feature Transformer**: Trainable compression of sparse input (3420 → 512)
- **Dual Accumulator**: 512×2 → 1024 (both perspectives concatenated)
- **Tiny Hidden Layers**: 16 → 32 → 1 (NOT wide — this is the key insight)
- **8 Output Buckets** (PSQTBuckets)
- **Quantized int16 inference** (quantize after training)

**~2M parameters** (mostly in feature transformer). Int16 quantized inference: ~1MB int16 weights for forward pass.

---

## 2. Architecture

### Fairy-Stockfish NNUE Structure

```
Sparse Input: ~77K features (128 active) → Feature Transformer → 512 compressed
                                                                        ↓
                                              1024 (512×2) → 16 → 32 → 1 → 8 buckets
```

For Xiangqi (3420 sparse features):
```
Input: [3420] (sparse binary, ~30 active)
  ↓ Feature Transformer (FC bottleneck): [3420] × [3420, 512]^T + [512] → [512]
  ↓ CReLU (clamp 0-1)
  │
  ├──► acc_red:   [512] → [16]  (CReLU)
  │
  └──► acc_black: [512] → [16]  (CReLU)
              │
         Concatenate
              │
              ▼
         [32] = [16 | 16]
              │
         Hidden2: [32] → [32] (CReLU)
              │
         Output: [32] → [1]
              │
         8 buckets (PSQTBuckets)
              │
         tanh × 400
Output: scalar ∈ [-400, 400]
```

### Parameter Count

| Layer | Weights | Biases |
|-------|---------|--------|
| Feature Transformer (3420→512) | 3420 × 512 = 1,751,040 | 512 |
| Dual Accumulator (512→16×2) | 512 × 16 × 2 = 16,384 | 16 × 2 = 32 |
| Hidden Layer (32→32) | 32 × 32 = 1,024 | 32 |
| Output (32→8) | 32 × 8 = 256 | 8 |
| **Total** | **1,768,704** | **580** |
| **Grand total** | **~1.77M parameters** | |

### Bucket System

8 buckets (PSQTBuckets), selected by non-king piece count:
- Formula: `clamp((count - 2) / 4, 0, 7)` (count: 0-30)
- Each bucket has its own output weights (32×8)

### Activation Functions

- **Feature Transformer**: ReLU
- **Accumulator outputs**: CReLU (clamp to [0, 1]) — same as NNUE
- **Hidden2**: CReLU (clamp to [0, 1])
- **Output**: Linear

### Quantization (Post-Training)

After training in FP32, quantize for inference:
- Feature transformer weights: int16 (QA = 127)
- Accumulator/output weights: int16 (QB = 64)
- Activations kept in int16 range [0, 127]
- Final output: int32 accumulation → divide by QB → tanh × 400 → f32

---

## 3. Training

### Framework
`burn` + `burn-ndarray` + `burn-autodiff` (FP32 training)

### Loss
MSE on bucket-selected value: `loss = mean((selected_bucket/400 - label/400)²)`

### Bucket Selection
- Supervised: bucket 7 (max material)
- Self-play: per-sample `non_king_count` → bucket index

### Quantization Strategy
Train in FP32 with burn. After training:
1. Convert weights to int16 (round to nearest)
2. Store feature transformer as int16 array
3. At inference: quantized int16 matmul → int32 accumulation → dequantize

For now: implement FP32 inference with option to add int16 quantization later.

---

## 4. Key Insight: Why Tiny Hidden Layers?

Fairy-Stockfish's architecture is counter-intuitive: **1024 → 16 → 32 → 1**. Most networks go wider, not narrower.

The reason: the feature transformer (input compression) already extracts the important information into 512 compressed features. The subsequent network just needs to do light mixing — hence the tiny layers. The expressiveness is in the **representation**, not the **width**.

This is why NNUE is fast on CPU: the expensive op is the int16 sparse→dense feature transformer (one matmul), not a large dense network.

---

## 5. Files to Modify

1. **`src/nn_eval.rs`**
   - Remove: CompactResNet, conv helpers, ResBlock, CompactResNetBurn
   - Add: `NNUEFeedForward` (plain ndarray, FP32)
   - Add: `NNUEFeedForwardBurn` (burn FP32 training)
   - Add: `bucket_index()` helper
   - Keep: `InputPlanes`, `NNOutput`, `nn_evaluate_or_handcrafted` (updated)

2. **`src/nn_train.rs`**
   - Add `non_king_count: u8` to `TrainingSample`
   - Update forward chains for new architecture
