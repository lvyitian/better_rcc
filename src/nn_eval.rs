// Neural network evaluation module
// Converts board state to input planes for NN inference

use crate::{Board, Color, PieceType};
#[allow(unused_imports)]
use crate::RuleSet;
use crate::eval::eval_impl::{handcrafted_evaluate, game_phase};
use ndarray::{Array4, Array3, Array2, Array1};

#[cfg(feature = "train")]
use burn::prelude::*;
#[cfg(feature = "train")]
use burn::module::Module;
#[cfg(feature = "train")]
use burn::nn::{PaddingConfig2d, Relu, BatchNorm, BatchNormConfig};

/// Input planes for neural network evaluation.
/// 38 planes of 9x10 = 3420 total values.
/// Layout: data[plane * 90 + y * 9 + x]
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InputPlanes {
    data: [f32; 38 * 90],
}

impl InputPlanes {
    /// Create input planes from board state, encoded from side_to_move's perspective.
    #[allow(dead_code)]
    pub fn from_board(board: &Board, side_to_move: Color) -> Self {
        let mut data = [0.0f32; 38 * 90];

        // Count duplicate pieces per side to assign correct plane index
        let mut red_advisors: usize = 0;
        let mut red_elephants: usize = 0;
        let mut red_horses: usize = 0;
        let mut red_cannons: usize = 0;
        let mut black_advisors: usize = 0;
        let mut black_elephants: usize = 0;
        let mut black_horses: usize = 0;
        let mut black_cannons: usize = 0;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.cells[y][x] {
                    // Determine if this piece is "ours" (side_to_move) or "theirs"
                    let our_base = match (piece.color, side_to_move) {
                        (Color::Red, Color::Red) | (Color::Black, Color::Black) => 0usize,
                        (Color::Red, Color::Black) | (Color::Black, Color::Red) => 14usize,
                    };

                    let base = our_base;
                    let plane = match piece.piece_type {
                        PieceType::King => base,
                        PieceType::Advisor => {
                            let idx = if piece.color == Color::Red {
                                let i = red_advisors; red_advisors += 1; i
                            } else {
                                let i = black_advisors; black_advisors += 1; i
                            };
                            base + 1 + idx.min(1)
                        }
                        PieceType::Elephant => {
                            let idx = if piece.color == Color::Red {
                                let i = red_elephants; red_elephants += 1; i
                            } else {
                                let i = black_elephants; black_elephants += 1; i
                            };
                            base + 3 + idx.min(1)
                        }
                        PieceType::Horse => {
                            let idx = if piece.color == Color::Red {
                                let i = red_horses; red_horses += 1; i
                            } else {
                                let i = black_horses; black_horses += 1; i
                            };
                            base + 5 + idx.min(1)
                        }
                        PieceType::Cannon => {
                            let idx = if piece.color == Color::Red {
                                let i = red_cannons; red_cannons += 1; i
                            } else {
                                let i = black_cannons; black_cannons += 1; i
                            };
                            base + 7 + idx.min(1)
                        }
                        PieceType::Pawn => {
                            // 5 pawns on files 0-4 (left to right)
                            let file = x.min(4);
                            base + 9 + file
                        }
                        PieceType::Chariot => {
                            // Chariots: own at plane 11, enemy at plane 25
                            if our_base == 0 { 11 } else { 25 }
                        }
                    };

                    data[plane * 90 + y * 9 + x] = 1.0;
                }
            }
        }

        // Plane 28: side-to-move indicator
        if side_to_move == Color::Red {
            for i in 0..90 { data[28 * 90 + i] = 1.0; }
        }

        // Plane 29: game phase (0=opening, 1=endgame)
        // game_phase returns 0 (all pieces gone=endgame) to 82 (full=opening)
        // Invert so that 0=opening and 1=endgame
        let phase = 1.0 - (game_phase(board) as f32 / 82.0f32).clamp(0.0, 1.0);
        for i in 0..90 { data[29 * 90 + i] = phase; }

        // Plane 30: repetition count (0-3+ normalized to 0-1)
        // Count how many times current position has occurred
        let reps = board.repetition_history
            .get(&board.zobrist_key)
            .copied()
            .unwrap_or(0) as f32;
        let reps_normalized = (reps.min(3.0) / 3.0).min(1.0);
        for i in 0..90 { data[30 * 90 + i] = reps_normalized; }

        // Plane 31: rule-60 counter approximated by move history length (0-90+ normalized to 0-1)
        // Uses total ply count from history as a proxy for halfmove clock
        let rule60 = (board.move_history.len().min(90) as f32 / 90.0).min(1.0);
        for i in 0..90 { data[31 * 90 + i] = rule60; }

        // Planes 32-33: placeholder zeros (already 0 from array init)

        Self { data }
    }

    /// Convert to ndarray Array4 with shape [1, 38, 9, 10].
    #[allow(dead_code)]
    pub fn to_array4(&self) -> ndarray::Array4<f32> {
        ndarray::Array4::from_shape_vec((1, 38, 9, 10), self.data.to_vec())
            .expect("InputPlanes shape [1, 38, 9, 10] is always valid")
    }

    /// Create from flat data.
    #[allow(dead_code)]
    pub fn from_flat(data: Vec<f32>) -> Self {
        let mut arr = [0.0f32; 3420];
        arr.copy_from_slice(&data);
        Self { data: arr }
    }

    /// Convert to flat Vec.
    #[allow(dead_code)]
    pub fn into_vec(self) -> Vec<f32> {
        self.data.to_vec()
    }

    /// Return reference to flat plane data for training data export.
    #[allow(dead_code)]
    pub fn planes_data(&self) -> &[f32; 3420] {
        &self.data
    }

    /// Return flat plane data as Vec for serialization.
    #[allow(dead_code)]
    pub fn planes_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    /// Convert to burn Tensor for training.
    #[cfg(feature = "train")]
    #[allow(dead_code)]
    pub fn to_burn_tensor<B: Backend>(&self) -> Tensor<B, 4> {
        use burn::tensor::TensorData;
        let device = <B as Backend>::Device::default();
        let data = TensorData::from(self.data);
        let flat: Tensor<B, 1> = Tensor::from_data(data, &device);
        flat.reshape([1, 38, 9, 10])
    }
}

// =============================================================================
// Compact ResNet for Neural Network Evaluation
// =============================================================================

/// He initialization for weights
#[allow(dead_code)]
fn he_init(_out_ch: usize, in_ch: usize, k: usize) -> f32 {
    let std = (2.0 / (in_ch as f32 * k as f32 * k as f32)).sqrt();
    (fast_rand() - 0.5) * 2.0 * std
}

/// Fast random number generator (Xorshift64)
#[allow(dead_code)]
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

/// Sigmoid with output range [min_val, max_val]
#[allow(dead_code)]
fn sigmoid_range(x: f32, min_val: f32, max_val: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    min_val + (max_val - min_val) * s
}

/// Naive 2D convolution: input [B, IC, H, W], weight [OC, IC, 3, 3], bias [OC, H, W]
#[allow(dead_code)]
fn conv2d(input: &Array4<f32>, weight: &Array4<f32>, bias: &Array3<f32>) -> Array4<f32> {
    let (batch, in_ch, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
    let (out_ch, _, k, _) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
    let pad = k / 2;
    let mut output = Array4::<f32>::zeros((batch, out_ch, h, w));

    for b in 0..batch {
        for oc in 0..out_ch {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = bias[[oc, y, x]];
                    for ic in 0..in_ch {
                        for ky in 0..k {
                            for kx in 0..k {
                                let in_y = y as i32 + ky as i32 - pad as i32;
                                let in_x = x as i32 + kx as i32 - pad as i32;
                                if in_y >= 0 && in_y < h as i32 && in_x >= 0 && in_x < w as i32 {
                                    sum += input[[b, ic, in_y as usize, in_x as usize]]
                                        * weight[[oc, ic, ky, kx]];
                                }
                            }
                        }
                    }
                    output[[b, oc, y, x]] = sum;
                }
            }
        }
    }
    output
}

/// ReLU activation
#[allow(dead_code)]
fn relu(x: &Array4<f32>) -> Array4<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Add two arrays of same shape (in-place via flat iteration)
#[allow(dead_code)]
fn add_arrays(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    let mut out = a.clone();
    let a_slice = out.as_slice_mut().unwrap();
    let b_slice = b.as_slice().unwrap();
    for i in 0..a_slice.len() {
        a_slice[i] += b_slice[i];
    }
    out
}

/// Global average pooling: [B, C, H, W] -> [B, C]
#[allow(dead_code)]
fn global_avg_pool(x: &Array4<f32>) -> Array2<f32> {
    let h = x.shape()[2] as f32;
    let w = x.shape()[3] as f32;
    let mut out = Array2::<f32>::zeros((x.shape()[0], x.shape()[1]));
    for b in 0..x.shape()[0] {
        for c in 0..x.shape()[1] {
            let mut sum = 0.0f32;
            for y in 0..x.shape()[2] {
                for x_idx in 0..x.shape()[3] {
                    sum += x[[b, c, y, x_idx]];
                }
            }
            out[[b, c]] = sum / (h * w);
        }
    }
    out
}

/// Compact ResNet for Chinese Chess evaluation.
/// ~180K parameters, optimized for CPU inference.
#[derive(Clone)]
#[allow(dead_code)]
pub struct CompactResNet {
    // Initial conv: 38 -> 32 channels
    init_w: Array4<f32>,
    init_b: Array3<f32>,

    // Block 1: 32 -> 32 (identity skip)
    b1_w1: Array4<f32>, b1_b1: Array3<f32>,
    b1_w2: Array4<f32>, b1_b2: Array3<f32>,

    // Block 2: 32 -> 32 (identity skip)
    b2_w1: Array4<f32>, b2_b1: Array3<f32>,
    b2_w2: Array4<f32>, b2_b2: Array3<f32>,

    // Block 3: 32 -> 64 (1x1 conv skip)
    b3_w1: Array4<f32>, b3_b1: Array3<f32>,
    b3_w2: Array4<f32>, b3_b2: Array3<f32>,
    b3_skip_w: Array4<f32>, b3_skip_b: Array3<f32>,

    // Block 4: 64 -> 64 (identity skip)
    b4_w1: Array4<f32>, b4_b1: Array3<f32>,
    b4_w2: Array4<f32>, b4_b2: Array3<f32>,

    // Block 5: 64 -> 64 (identity skip)
    b5_w1: Array4<f32>, b5_b1: Array3<f32>,
    b5_w2: Array4<f32>, b5_b2: Array3<f32>,

    // Block 6: 64 -> 64 (identity skip)
    b6_w1: Array4<f32>, b6_b1: Array3<f32>,
    b6_w2: Array4<f32>, b6_b2: Array3<f32>,

    // Dense: 64 -> 64
    dense_w: Array2<f32>, dense_b: Array1<f32>,

    // Output heads: 64 -> 1 each
    alpha_w: Array2<f32>, alpha_b: Array1<f32>,
    beta_w: Array2<f32>, beta_b: Array1<f32>,
    score_w: Array2<f32>, score_b: Array1<f32>,
    correction_w: Array2<f32>, correction_b: Array1<f32>,
}

#[allow(dead_code)]
impl CompactResNet {
    pub fn new() -> Self {
        let make4 = |oc, ic, k: usize| -> Array4<f32> {
            let mut a = Array4::<f32>::zeros((oc, ic, k, k));
            for i in 0..oc {
                for j in 0..ic {
                    for ki in 0..k {
                        for kj in 0..k {
                            a[[i, j, ki, kj]] = he_init(oc, ic, k);
                        }
                    }
                }
            }
            a
        };
        let make3 = |c, h, w| Array3::<f32>::zeros((c, h, w));
        let make2 = |r, c| Array2::<f32>::zeros((r, c));
        let make1 = |n| Array1::<f32>::zeros(n);

        Self {
            init_w: make4(32, 38, 3), init_b: make3(32, 9, 10),
            b1_w1: make4(32, 32, 3), b1_b1: make3(32, 9, 10),
            b1_w2: make4(32, 32, 3), b1_b2: make3(32, 9, 10),
            b2_w1: make4(32, 32, 3), b2_b1: make3(32, 9, 10),
            b2_w2: make4(32, 32, 3), b2_b2: make3(32, 9, 10),
            b3_w1: make4(64, 32, 3), b3_b1: make3(64, 9, 10),
            b3_w2: make4(64, 64, 3), b3_b2: make3(64, 9, 10),
            b3_skip_w: make4(64, 32, 1), b3_skip_b: make3(64, 9, 10),
            b4_w1: make4(64, 64, 3), b4_b1: make3(64, 9, 10),
            b4_w2: make4(64, 64, 3), b4_b2: make3(64, 9, 10),
            b5_w1: make4(64, 64, 3), b5_b1: make3(64, 9, 10),
            b5_w2: make4(64, 64, 3), b5_b2: make3(64, 9, 10),
            b6_w1: make4(64, 64, 3), b6_b1: make3(64, 9, 10),
            b6_w2: make4(64, 64, 3), b6_b2: make3(64, 9, 10),
            dense_w: make2(64, 64), dense_b: make1(64),
            alpha_w: make2(1, 64), alpha_b: make1(1),
            beta_w: make2(1, 64), beta_b: make1(1),
            score_w: make2(1, 64), score_b: make1(1),
            correction_w: make2(1, 64), correction_b: make1(1),
        }
    }

    /// Residual block forward
    fn res_block(&self, x: &Array4<f32>,
                 w1: &Array4<f32>, b1: &Array3<f32>,
                 w2: &Array4<f32>, b2: &Array3<f32>,
                 skip: Option<(&Array4<f32>, &Array3<f32>)>) -> Array4<f32> {
        let h = relu(&conv2d(x, w1, b1));
        let h = conv2d(&h, w2, b2);
        let skip_val = match skip {
            Some((sw, sb)) => conv2d(x, sw, sb),
            None => x.clone(),
        };
        add_arrays(&h, &skip_val)
    }

    pub fn forward(&self, input: &InputPlanes) -> NNOutput {
        // Initial conv: 38 -> 32, ReLU
        let x = relu(&conv2d(&input.to_array4(), &self.init_w, &self.init_b));

        // Block 1: 32 -> 32
        let x = self.res_block(&x, &self.b1_w1, &self.b1_b1, &self.b1_w2, &self.b1_b2, None);

        // Block 2: 32 -> 32
        let x = self.res_block(&x, &self.b2_w1, &self.b2_b1, &self.b2_w2, &self.b2_b2, None);

        // Block 3: 32 -> 64 (downsample)
        let x = self.res_block(&x, &self.b3_w1, &self.b3_b1, &self.b3_w2, &self.b3_b2,
                               Some((&self.b3_skip_w, &self.b3_skip_b)));

        // Block 4: 64 -> 64
        let x = self.res_block(&x, &self.b4_w1, &self.b4_b1, &self.b4_w2, &self.b4_b2, None);

        // Block 5: 64 -> 64
        let x = self.res_block(&x, &self.b5_w1, &self.b5_b1, &self.b5_w2, &self.b5_b2, None);

        // Block 6: 64 -> 64
        let x = self.res_block(&x, &self.b6_w1, &self.b6_b1, &self.b6_w2, &self.b6_b2, None);

        // Global average pooling: [1, 64, 9, 10] -> [1, 64]
        let pooled = global_avg_pool(&x);

        // Dense: 64 -> 64, ReLU
        let mut dense = pooled.dot(&self.dense_w.t());
        {
            let slice = dense.as_slice_mut().unwrap();
            for v in slice.iter_mut() { *v = v.max(0.0); }
        }

        // Heads
        let alpha_raw = dense[[0, 0]] * self.alpha_w[[0, 0]] + self.alpha_b[[0]];
        let beta_raw = dense[[0, 0]] * self.beta_w[[0, 0]] + self.beta_b[[0]];
        let score_raw = dense[[0, 0]] * self.score_w[[0, 0]] + self.score_b[[0]];
        let correction_raw = dense[[0, 0]] * self.correction_w[[0, 0]] + self.correction_b[[0]];

        let alpha = sigmoid_range(alpha_raw, 0.05, 0.95);
        let beta = sigmoid_range(beta_raw, 0.05, 0.95);
        let nn_score = score_raw.tanh() * 300.0;
        let correction = correction_raw.tanh() * 300.0;

        NNOutput { alpha, beta, nn_score, correction }
    }
}

impl Default for CompactResNet {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// Compact ResNet for Neural Network Evaluation (burn)
// =============================================================================

/// Residual block for burn: two 3×3 convs with optional 1×1 skip for channel changes.
/// Per ResNet v2: bn → relu → conv for each conv layer; skip added before final relu.
#[cfg(feature = "train")]
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    pub conv1: burn::nn::conv::Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub conv2: burn::nn::conv::Conv2d<B>,
    pub bn2: BatchNorm<B>,
    /// Optional 1×1 conv for channel change in skip connection.
    pub skip_conv: Option<burn::nn::conv::Conv2d<B>>,
    /// Optional 1×1 conv to expand channels before bn1 when in_ch != out_ch.
    pub expand_conv: Option<burn::nn::conv::Conv2d<B>>,
    pub relu: Relu,
}

#[cfg(feature = "train")]
impl<B: Backend> ResBlock<B> {
    /// channels: [in, out]. If in != out, creates a 1×1 conv skip.
    fn new(channels: [usize; 2]) -> Self {
        let device = <B as Backend>::Device::default();
        let (in_ch, out_ch) = (channels[0], channels[1]);

        let bn1 = BatchNormConfig::new(out_ch)
            .with_momentum(0.1)
            .with_epsilon(1e-5)
            .init(&device);
        let conv1 = burn::nn::conv::Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(true)
            .init(&device);

        let bn2 = BatchNormConfig::new(out_ch)
            .with_momentum(0.1)
            .with_epsilon(1e-5)
            .init(&device);
        let conv2 = burn::nn::conv::Conv2dConfig::new([out_ch, out_ch], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(true)
            .init(&device);

        let skip_conv = if in_ch != out_ch {
            Some(burn::nn::conv::Conv2dConfig::new([in_ch, out_ch], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(true)
                .init(&device))
        } else {
            None
        };

        // Expand conv: 1x1 conv to expand channels before bn1 when in_ch != out_ch.
        // This ensures bn1 (configured with out_ch) receives a tensor with out_ch channels.
        let expand_conv = if in_ch != out_ch {
            Some(burn::nn::conv::Conv2dConfig::new([in_ch, out_ch], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(true)
                .init(&device))
        } else {
            None
        };

        Self { conv1, bn1, conv2, bn2, skip_conv, expand_conv, relu: Relu::new() }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // ResNet v2: bn → relu → conv for each conv layer.
        // Main path: bn1 → relu → conv1 → bn2 → relu → conv2
        // Skip path: 1x1 conv on raw x (when channels differ), or identity
        // Final: add + relu

        // Compute skip from raw x — no bn on skip path in ResNet v2
        let skip_h = if let Some(ref sc) = self.skip_conv {
            sc.forward(x.clone())
        } else {
            x.clone()
        };

        // Main path: expand channels (if needed) → bn1 → relu → conv1
        // When in_ch != out_ch, expand_conv expands x to out_ch channels BEFORE bn1,
        // ensuring bn1 (configured with out_ch) receives a tensor with matching channels.
        let h = if let Some(ref ec) = self.expand_conv {
            ec.forward(x)
        } else {
            x
        };
        let mut h = self.bn1.forward(h);
        h = self.relu.forward(h);
        h = self.conv1.forward(h);

        // Main path: bn2 → relu → conv2
        h = self.bn2.forward(h);
        h = self.relu.forward(h);
        h = self.conv2.forward(h);

        // Final add + relu
        let h = h + skip_h;
        self.relu.forward(h)
    }
}

/// Compact ResNet using burn Module derive.
#[cfg(feature = "train")]
#[derive(Module, Debug)]
pub struct CompactResNetBurn<B: Backend = burn_ndarray::NdArray<f32>> {
    pub init_conv: burn::nn::conv::Conv2d<B>,
    pub b1: ResBlock<B>,
    pub b2: ResBlock<B>,
    pub b3: ResBlock<B>,
    pub b4: ResBlock<B>,
    pub b5: ResBlock<B>,
    pub b6: ResBlock<B>,
    pub dense: burn::nn::Linear<B>,
    pub alpha_head: burn::nn::Linear<B>,
    pub beta_head: burn::nn::Linear<B>,
    pub score_head: burn::nn::Linear<B>,
    pub correction_head: burn::nn::Linear<B>,
    pub relu: Relu,
}

#[cfg(feature = "train")]
impl<B: Backend> CompactResNetBurn<B> {
    pub fn new() -> Self {
        let device = <B as Backend>::Device::default();
        let init_conv = burn::nn::conv::Conv2dConfig::new([38, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(true)
            .init(&device);
        let b1 = ResBlock::new([32, 32]);
        let b2 = ResBlock::new([32, 32]);
        let b3 = ResBlock::new([32, 64]); // downsample: 32→64
        let b4 = ResBlock::new([64, 64]);
        let b5 = ResBlock::new([64, 64]);
        let b6 = ResBlock::new([64, 64]);
        let dense = burn::nn::LinearConfig::new(64, 64)
            .with_bias(true)
            .init(&device);
        let alpha_head = burn::nn::LinearConfig::new(64, 1)
            .with_bias(true)
            .init(&device);
        let beta_head = burn::nn::LinearConfig::new(64, 1)
            .with_bias(true)
            .init(&device);
        let score_head = burn::nn::LinearConfig::new(64, 1)
            .with_bias(true)
            .init(&device);
        let correction_head = burn::nn::LinearConfig::new(64, 1)
            .with_bias(true)
            .init(&device);
        Self {
            init_conv, b1, b2, b3, b4, b5, b6,
            dense, alpha_head, beta_head, score_head, correction_head,
            relu: Relu::new(),
        }
    }

    /// Forward pass producing NNOutput.
    pub fn forward_for_inference(&self, planes: &InputPlanes) -> NNOutput
    {
        use burn::tensor::TensorData;
        let device = <B as Backend>::Device::default();
        // Convert flat [f32; 3420] → Tensor<B, 4> shape [1, 38, 9, 10]
        let flat_arr: [f32; 3420] = planes.data;
        let data = TensorData::from(flat_arr);
        let flat: Tensor<B, 1> = Tensor::from_data(data, &device);
        let x: Tensor<B, 4> = flat.reshape([1, 38, 9, 10]);

        // Initial conv: 38 → 32
        let x = self.relu.forward(self.init_conv.forward(x));

        // Residual blocks
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        let x = self.b3.forward(x);
        let x = self.b4.forward(x);
        let x = self.b5.forward(x);
        let x = self.b6.forward(x);

        // Global average pool: [1, 64, 9, 10] → [1, 64]
        let pooled = x.mean_dim(3).mean_dim(2).reshape([1, 64]); // [1, 64]

        // Dense + ReLU
        let dense_out = self.relu.forward(self.dense.forward(pooled));

        // Heads
        let to_f32 = |t: Tensor<B, 2>| -> f32 {
            t.to_data().as_slice().expect("expected 1x1 tensor")[0]
        };
        let alpha_raw = to_f32(self.alpha_head.forward(dense_out.clone()));
        let beta_raw = to_f32(self.beta_head.forward(dense_out.clone()));
        let score_raw = to_f32(self.score_head.forward(dense_out.clone()));
        let correction_raw = to_f32(self.correction_head.forward(dense_out));

        let alpha = (1.0f32 / (1.0f32 + (-alpha_raw).exp())) * 0.9 + 0.05;
        let beta = (1.0f32 / (1.0f32 + (-beta_raw).exp())) * 0.9 + 0.05;
        let nn_score = score_raw.tanh() * 300.0;
        let correction = correction_raw.tanh() * 300.0;

        NNOutput { alpha, beta, nn_score, correction }
    }
}

#[cfg(feature = "train")]
impl<B: Backend> Default for CompactResNetBurn<B> {
    fn default() -> Self { Self::new() }
}

/// Output of the neural network forward pass.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct NNOutput {
    /// Alpha weight for NN score component. Range [0.05, 0.95].
    pub alpha: f32,
    /// Beta weight for handcrafted score component. Range [0.05, 0.95].
    pub beta: f32,
    /// NN raw score in centipawns. Range [-300, 300].
    pub nn_score: f32,
    /// Additive correction. Range [-300, 300] centipawns.
    pub correction: f32,
}

/// Global NN instance (ndarray version), lazily initialized.
#[cfg(not(feature = "train"))]
static NN_NET: std::sync::LazyLock<CompactResNet> =
    std::sync::LazyLock::new(CompactResNet::new);

/// Hybrid NN + handcrafted evaluation.
///
/// Blending formula (per spec):
///   final_score = (alpha / (alpha + beta)) * nn_score
///               + (beta / (alpha + beta)) * handcrafted_score
///               + correction
///
/// Where alpha, beta ∈ [0.05, 0.95] (sigmoid-clamped), nn_score ∈ [-300, 300],
/// correction ∈ [-300, 300], handcrafted_score is in centipawns.
#[allow(dead_code)]
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);

    let input = InputPlanes::from_board(board, side);

    #[cfg(feature = "train")]
    let output = {
        type InferenceBackend = burn_ndarray::NdArray<f32>;
        static NET: std::sync::LazyLock<CompactResNetBurn<InferenceBackend>> =
            std::sync::LazyLock::new(CompactResNetBurn::new);
        NET.forward_for_inference(&input)
    };

    #[cfg(not(feature = "train"))]
    let output = NN_NET.forward(&input);

    // Normalize alpha + beta to sum to 1 (with minimum floor of 0.05)
    let alpha = output.alpha.max(0.05);
    let beta = output.beta.max(0.05);
    let total = alpha + beta;
    let alpha_norm = alpha / total;
    let beta_norm = beta / total;

    // NN outputs already scaled to centipawns
    let nn_score = output.nn_score;
    // Correction is in [-300, 300] range, use as additive centipawn offset
    let correction = output.correction;

    let blended = alpha_norm * nn_score + beta_norm * handcrafted as f32 + correction;
    blended as i32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_planes_from_board_starts_position() {
        // Should not panic on starting position
        let board = Board::new(RuleSet::Official, 1);
        let planes = InputPlanes::from_board(&board, Color::Red);
        // Verify all 3420 values are in valid range [0, 1]
        for v in &planes.data {
            assert!((0.0..=1.0).contains(v), "plane value {} out of range [0,1]", v);
        }
    }

    #[test]
    fn test_input_planes_to_array4_shape() {
        let board = Board::new(RuleSet::Official, 1);
        let planes = InputPlanes::from_board(&board, Color::Red);
        let arr = planes.to_array4();
        assert_eq!(arr.shape(), &[1, 38, 9, 10]);
    }

    #[test]
    fn test_input_planes_sidemove_plane() {
        let board = Board::new(RuleSet::Official, 1);
        let planes_red = InputPlanes::from_board(&board, Color::Red);
        let planes_black = InputPlanes::from_board(&board, Color::Black);
        // Red side-to-move plane (28) should be all 1s
        for i in 0..90 {
            assert_eq!(planes_red.data[28 * 90 + i], 1.0, "Red stm plane[{}] should be 1.0", i);
        }
        // Black side-to-move plane (28) should be all 0s
        for i in 0..90 {
            assert_eq!(planes_black.data[28 * 90 + i], 0.0, "Black stm plane[{}] should be 0.0", i);
        }
    }

    #[test]
    fn test_compact_resnet_forward_output_ranges() {
        let net = CompactResNet::new();
        let board = Board::new(RuleSet::Official, 1);
        let planes = InputPlanes::from_board(&board, Color::Red);
        let output = net.forward(&planes);

        // Alpha and beta should be in [0.05, 0.95]
        assert!(output.alpha >= 0.05 && output.alpha <= 0.95,
            "alpha {} out of range [0.05, 0.95]", output.alpha);
        assert!(output.beta >= 0.05 && output.beta <= 0.95,
            "beta {} out of range [0.05, 0.95]", output.beta);

        // NN score and correction should be in roughly [-300, 300]
        assert!(output.nn_score >= -300.0 && output.nn_score <= 300.0,
            "nn_score {} out of range [-300, 300]", output.nn_score);
        assert!(output.correction >= -300.0 && output.correction <= 300.0,
            "correction {} out of range [-300, 300]", output.correction);
    }

    #[test]
    fn test_nnoutput_debug() {
        let out = NNOutput {
            alpha: 0.5,
            beta: 0.5,
            nn_score: 100.0,
            correction: 10.0,
        };
        let debug_str = format!("{:?}", out);
        assert!(debug_str.contains("alpha"));
        assert!(debug_str.contains("nn_score"));
    }
}
