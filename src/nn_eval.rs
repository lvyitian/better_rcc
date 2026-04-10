// Neural network evaluation module
// Converts board state to input planes for NN inference

use crate::{Board, Color, PieceType};
#[allow(unused_imports)]
use crate::RuleSet;
use crate::eval::eval::{handcrafted_evaluate, game_phase};
use ndarray::{Array4, Array3, Array2, Array1};

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
                        PieceType::King => base + 0,
                        PieceType::Advisor => {
                            let idx = if piece.color == Color::Red {
                                let i = red_advisors; red_advisors += 1; i
                            } else {
                                let i = black_advisors; black_advisors += 1; i
                            };
                            base + 1 + idx.min(1) as usize
                        }
                        PieceType::Elephant => {
                            let idx = if piece.color == Color::Red {
                                let i = red_elephants; red_elephants += 1; i
                            } else {
                                let i = black_elephants; black_elephants += 1; i
                            };
                            base + 3 + idx.min(1) as usize
                        }
                        PieceType::Horse => {
                            let idx = if piece.color == Color::Red {
                                let i = red_horses; red_horses += 1; i
                            } else {
                                let i = black_horses; black_horses += 1; i
                            };
                            base + 5 + idx.min(1) as usize
                        }
                        PieceType::Cannon => {
                            let idx = if piece.color == Color::Red {
                                let i = red_cannons; red_cannons += 1; i
                            } else {
                                let i = black_cannons; black_cannons += 1; i
                            };
                            base + 7 + idx.min(1) as usize
                        }
                        PieceType::Pawn => {
                            // 5 pawns on files 0-4 (left to right)
                            let file = (x as usize).min(4);
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
        let phase = (game_phase(board) as f32 / 82.0f32).min(1.0).max(0.0);
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
// Evaluation Dispatch
// =============================================================================

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

/// Global NN instance, lazily initialized.
static NN_NET: std::sync::LazyLock<CompactResNet> =
    std::sync::LazyLock::new(CompactResNet::new);

/// Hybrid NN + handcrafted evaluation.
///
/// Blending formula (per spec):
///   final_score = (alpha / (alpha + beta)) * nn_score
///               + (beta / (alpha + beta)) * handcrafted_score
///               + correction * scale
///
/// Where alpha, beta ∈ [0.05, 0.95] (sigmoid-clamped), nn_score ∈ [-300, 300],
/// correction ∈ [-300, 300], handcrafted_score is in centipawns.
#[allow(dead_code)]
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);

    let input = InputPlanes::from_board(board, side);
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
        for &v in &planes.data {
            assert!(v >= 0.0 && v <= 1.0, "plane value {} out of range [0,1]", v);
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
