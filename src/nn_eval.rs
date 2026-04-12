// Neural network evaluation module
// Converts board state to input planes for NN inference

use crate::{Board, Color, PieceType, MATE_SCORE};
#[allow(unused_imports)]
use crate::RuleSet;
use crate::eval::eval_impl::{handcrafted_evaluate, game_phase};
#[cfg(feature = "train")]
use burn::prelude::*;
#[cfg(feature = "train")]
use burn::module::Module;

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
// NNUE (Efficiently Updatable Neural Networks) Evaluation
// Bullet-compliant (1260→1024)×2→8 architecture
// =============================================================================

use crate::nnue_input::{INPUT_DIM, FT_DIM, NUM_BUCKETS, QA, QB, SCALE, bucket_index, NNInputPlanes};

/// Feature transform output: 1024 i16 values, cache-line aligned for SIMD.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Accumulator {
    pub vals: [i16; FT_DIM],
}

impl serde::Serialize for Accumulator {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.vals.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for Accumulator {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = [i16; FT_DIM];
            fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "an array of {} i16 values", FT_DIM)
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut arr = [0i16; FT_DIM];
                for i in 0..FT_DIM {
                    arr[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::custom("too few elements"))?;
                }
                Ok(arr)
            }
        }
        let vals = de.deserialize_seq(V)?;
        Ok(Accumulator { vals })
    }
}

impl Accumulator {
    fn zero() -> Self {
        Self { vals: [0i16; FT_DIM] }
    }
}

/// SCReLU activation: clamp(x, 0, 255)² returning i32.
#[inline]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, 255);
    y * y
}

/// NNUE feed-forward network using plain ndarray (inference only).
#[allow(dead_code)]
pub struct NNUEFeedForward {
    /// Feature transform weights: [INPUT_DIM][FT_DIM] stored as 1260 Accumulators (column-major).
    /// Each column f has weights for feature f across all 1024 hidden units.
    pub ft_weights: Vec<Accumulator>,
    /// Feature transform bias: [FT_DIM], QA quantized.
    pub ft_bias: [i16; FT_DIM],
    /// Output layer weights: [NUM_BUCKETS][FT_DIM * 2] for bucketized output.
    /// Each bucket has 2048 = 1024 (stm) + 1024 (ntm) inputs.
    pub out_weights: [[i16; FT_DIM * 2]; NUM_BUCKETS],
    /// Output layer bias: [NUM_BUCKETS], QA * QB quantized.
    pub out_bias: [i16; NUM_BUCKETS],
}

impl NNUEFeedForward {
    /// Create with zero-initialized weights.
    pub fn new() -> Self {
        Self {
            ft_weights: vec![Accumulator::zero(); INPUT_DIM],
            ft_bias: [0i16; FT_DIM],
            out_weights: [[0i16; FT_DIM * 2]; NUM_BUCKETS],
            out_bias: [0i16; NUM_BUCKETS],
        }
    }

    /// Initialize with random weights for testing.
    #[allow(dead_code)]
    pub fn random() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static STATE: AtomicU64 = AtomicU64::new(0x123456789ABCDEF0);
        fn next_u64() -> u64 {
            let current = STATE.load(Ordering::Relaxed);
            let new_val = current.wrapping_mul(6364136223846793005).wrapping_add(1);
            STATE.store(new_val, Ordering::Relaxed);
            new_val
        }
        fn rand_i16() -> i16 {
            (next_u64() % (QA as u64 * 2)) as i16 - QA as i16
        }

        let mut ft_weights = Vec::with_capacity(INPUT_DIM);
        for _ in 0..INPUT_DIM {
            ft_weights.push(Accumulator { vals: [rand_i16(); FT_DIM] });
        }
        let ft_bias = [rand_i16(); FT_DIM];
        let out_weights = [[rand_i16(); FT_DIM * 2]; NUM_BUCKETS];
        let out_bias = [rand_i16(); NUM_BUCKETS];

        Self { ft_weights, ft_bias, out_weights, out_bias }
    }

    /// Serialize this network to a byte vector using bincode.
    /// Flattens arrays into Vecs for compatibility with bincode.
    #[allow(dead_code)]
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        // Flatten ft_weights: [INPUT_DIM][FT_DIM] → Vec<i16>
        let ft_weights_flat: Vec<i16> = self
            .ft_weights
            .iter()
            .flat_map(|acc| acc.vals.iter().copied())
            .collect();
        // Flatten out_weights: [[i16; FT_DIM*2]; NUM_BUCKETS] → Vec<i16>
        let out_weights_flat: Vec<i16> = self
            .out_weights
            .iter()
            .flat_map(|bucket| bucket.iter().copied())
            .collect();
        let data = (
            ft_weights_flat,
            self.ft_bias.to_vec(),
            out_weights_flat,
            self.out_bias.to_vec(),
        );
        bincode::serialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Deserialize network from a byte vector.
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let (ft_weights_flat, ft_bias, out_weights_flat, out_bias): (
            Vec<i16>,
            Vec<i16>,
            Vec<i16>,
            Vec<i16>,
        ) = bincode::deserialize(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Reconstruct ft_weights: Vec<i16> → Vec<Accumulator>
        if ft_weights_flat.len() != INPUT_DIM * FT_DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ft_weights_flat size {} != {}*{}",
                    ft_weights_flat.len(),
                    INPUT_DIM,
                    FT_DIM
                ),
            ));
        }
        let mut ft_weights = Vec::with_capacity(INPUT_DIM);
        for f in 0..INPUT_DIM {
            let mut vals = [0i16; FT_DIM];
            for i in 0..FT_DIM {
                vals[i] = ft_weights_flat[f * FT_DIM + i];
            }
            ft_weights.push(Accumulator { vals });
        }

        if ft_bias.len() != FT_DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ft_bias size {} != FT_DIM {}", ft_bias.len(), FT_DIM),
            ));
        }
        let mut ft_bias_arr = [0i16; FT_DIM];
        ft_bias_arr.copy_from_slice(&ft_bias);

        // Reconstruct out_weights: Vec<i16> → [[i16; FT_DIM*2]; NUM_BUCKETS]
        if out_weights_flat.len() != NUM_BUCKETS * FT_DIM * 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "out_weights_flat size {} != {}*{}*2",
                    out_weights_flat.len(),
                    NUM_BUCKETS,
                    FT_DIM
                ),
            ));
        }
        let mut out_weights = [[0i16; FT_DIM * 2]; NUM_BUCKETS];
        for b in 0..NUM_BUCKETS {
            for i in 0..FT_DIM * 2 {
                out_weights[b][i] = out_weights_flat[b * FT_DIM * 2 + i];
            }
        }

        if out_bias.len() != NUM_BUCKETS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("out_bias size {} != NUM_BUCKETS {}", out_bias.len(), NUM_BUCKETS),
            ));
        }
        let mut out_bias_arr = [0i16; NUM_BUCKETS];
        out_bias_arr.copy_from_slice(&out_bias);

        Ok(NNUEFeedForward {
            ft_weights,
            ft_bias: ft_bias_arr,
            out_weights,
            out_bias: out_bias_arr,
        })
    }

    /// Load network from a binary file.
    /// First tries zstd decompression (new format), falls back to raw bincode (old format).
    /// Prints a warning and falls back to random if the file doesn't exist or fails to load.
    pub fn load_from_file(path: &str) -> Self {
        match Self::from_file_impl(path) {
            Ok(net) => net,
            Err(e) => {
                eprintln!(
                    "[NNUE] WARNING: failed to load network from '{}': {e}; using random weights",
                    path
                );
                Self::random()
            }
        }
    }

    pub fn from_file_impl(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Try zstd decompression first (new format with 4-byte raw size header)
        if data.len() >= 4 {
            let raw_size = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
            let compressed = &data[4..];
            if let Ok(decompressed) = zstd::decode_all(compressed) {
                if decompressed.len() == raw_size {
                    eprintln!("[NNUE] loaded weights from '{}' (zstd)", path);
                    return Self::from_bytes(&decompressed);
                }
            }
        }

        // Fall back to raw bincode (no compression, e.g. old saves)
        eprintln!("[NNUE] loaded weights from '{}' (raw)", path);
        Self::from_bytes(&data)
    }

    /// Serialize and zstd-compress to a file.
    /// Format on disk: [u32 raw_size][zstd compressed bincode(data)].
    #[allow(dead_code)]
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let raw = self.to_bytes()?;
        let compressed = zstd::encode_all(raw.as_slice(), 3)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let raw_size = raw.len() as u32;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&raw_size.to_le_bytes())?;
        file.write_all(&compressed)?;
        file.flush()?;
        Ok(())
    }

    /// Compute accumulators for stm and ntm inputs.
    /// Each accumulator = ft_bias + sum over active features of ft_weights[feature].
    /// Result clamped to [0, QA] and stored as i16.
    fn compute_accumulators(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM]) -> (Accumulator, Accumulator) {
        let mut stm_acc = Accumulator::zero();
        let mut ntm_acc = Accumulator::zero();

        // Add ft_bias to accumulators
        for i in 0..FT_DIM {
            stm_acc.vals[i] = self.ft_bias[i];
            ntm_acc.vals[i] = self.ft_bias[i];
        }

        // Add weights for active features
        for f in 0..INPUT_DIM {
            if stm[f] != 0.0 {
                for i in 0..FT_DIM {
                    stm_acc.vals[i] = stm_acc.vals[i].saturating_add(self.ft_weights[f].vals[i]);
                }
            }
            if ntm[f] != 0.0 {
                for i in 0..FT_DIM {
                    ntm_acc.vals[i] = ntm_acc.vals[i].saturating_add(self.ft_weights[f].vals[i]);
                }
            }
        }

        // Clamp to [0, QA] and convert to i16
        for i in 0..FT_DIM {
            stm_acc.vals[i] = stm_acc.vals[i].clamp(0, QA as i16);
            ntm_acc.vals[i] = ntm_acc.vals[i].clamp(0, QA as i16);
        }

        (stm_acc, ntm_acc)
    }

    /// Forward pass returning raw f32 score.
    /// `non_king_count` determines which bucket to use.
    #[allow(dead_code)]
    pub fn forward(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM], non_king_count: u8) -> f32 {
        let (stm_acc, ntm_acc) = self.compute_accumulators(stm, ntm);

        // Apply SCReLU: clamp(0, QA)² → i32
        // Then split and concatenate: [stm_acc(1024), ntm_acc(1024)] = 2048
        let mut combined = [0i32; FT_DIM * 2];
        for i in 0..FT_DIM {
            combined[i] = screlu(stm_acc.vals[i]);
            combined[FT_DIM + i] = screlu(ntm_acc.vals[i]);
        }

        // Select bucket
        let bucket_idx = bucket_index(non_king_count);

        // Compute dot product: out_weights[bucket] · combined
        // out_weights are QB quantized, combined is i32 (SCReLU result)
        // combined[i] = screlu(acc[i]) where screlu returns [0, 65025] as i32
        let mut raw = 0i64;
        #[allow(clippy::needless_range_loop)]
        for i in 0..FT_DIM * 2 {
            raw += i64::from(self.out_weights[bucket_idx][i]) * i64::from(combined[i]);
        }

        // Spec formula: output = ((Σ/QA) + bias) * SCALE / (QA*QB) then tanh * SCALE
        // Σ/QA reduces quantization from QA²·QB → QA·QB
        // Bias is stored in QA*QB units, so no conversion needed when adding
        let raw_f = raw as f32;
        let qb_f = QB as f32;
        let qa_f = QA as f32;
        let scale_f = SCALE as f32;

        // Divide by QA first (reduces QA²·QB → QA·QB), then add bias (QA*QB units)
        // Then apply tanh * SCALE to get final output in [-400, 400]
        let raw_result = ((raw_f / qa_f) + f32::from(self.out_bias[bucket_idx])) * scale_f / (qa_f * qb_f);
        raw_result.tanh() * scale_f
    }

    /// Forward pass returning NNOutput.
    #[allow(dead_code)]
    pub fn forward_output(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM], non_king_count: u8) -> NNOutput {
        let score = self.forward(stm, ntm, non_king_count);
        // For NNUE, alpha=1.0, beta=0.0 (pure NN evaluation)
        NNOutput {
            alpha: 1.0,
            beta: 0.0,
            nn_score: score,
            correction: 0.0,
        }
    }
}

impl Default for NNUEFeedForward {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// NNUE Feed-Forward with Burn (training)
// =============================================================================

#[cfg(feature = "train")]
#[allow(dead_code)]
#[derive(Module, Debug)]
pub struct NNUEFeedForwardBurn<B: Backend = burn_ndarray::NdArray<f32>> {
    /// Feature transform: 1260 → 1024 linear layer
    pub ft: burn::nn::Linear<B>,
    /// Output: 2048 → 8 bucket linear layer
    pub out: burn::nn::Linear<B>,
}

#[cfg(feature = "train")]
impl<B: Backend> NNUEFeedForwardBurn<B> {
    pub fn new() -> Self {
        let device = <B as Backend>::Device::default();
        // Feature transform: 1260 → 1024
        let ft = burn::nn::LinearConfig::new(INPUT_DIM, FT_DIM)
            .with_bias(true)
            .init(&device);
        // Output: 2048 → 8 buckets
        let out = burn::nn::LinearConfig::new(FT_DIM * 2, NUM_BUCKETS)
            .with_bias(true)
            .init(&device);
        Self { ft, out }
    }

    /// Forward with explicit bucket index (for single sample).
    /// Accepts 1D tensors [1260] for single-sample evaluation.
    pub fn forward_with_bucket(&self, stm: &Tensor<B, 1>, ntm: &Tensor<B, 1>, bucket_idx: usize) -> Tensor<B, 1> {
        // Reshape to [1, 1260], forward batch, squeeze back to scalar
        let stm_2d = stm.clone().reshape([1, INPUT_DIM]);
        let ntm_2d = ntm.clone().reshape([1, INPUT_DIM]);
        let result = self.forward_batched_impl(&stm_2d, &ntm_2d, bucket_idx);
        result.reshape([1])
    }

    /// Forward returning full [8] bucket outputs for a single sample.
    /// [1, 1260] → [8] (all bucket values before tanh).
    #[allow(dead_code)]
    pub fn forward_all_buckets(&self, stm: &Tensor<B, 1>, ntm: &Tensor<B, 1>) -> Tensor<B, 1> {
        let stm_2d = stm.clone().reshape([1, INPUT_DIM]);
        let ntm_2d = ntm.clone().reshape([1, INPUT_DIM]);

        // FT: [1, 1260] → [1, 1024]
        let ft_stm = self.ft.forward(stm_2d);
        let ft_ntm = self.ft.forward(ntm_2d);

        // SCReLU: clamp(0, 255)²
        let clamped_stm = ft_stm.clamp(0.0, 255.0);
        let clamped_ntm = ft_ntm.clamp(0.0, 255.0);
        let stm_act = clamped_stm.clone() * clamped_stm;
        let ntm_act = clamped_ntm.clone() * clamped_ntm;

        // Concatenate: [1, 2048]
        let combined = Tensor::cat(vec![stm_act, ntm_act], 1);

        // Output: [1, 2048] → [1, 8]
        self.out.forward(combined).reshape([8])
    }

    /// Batched forward pass: [batch, 1260] → [batch] bucket-selected scalar output.
    /// Applies FT → SCReLU → concat → output projection → gather bucket.
    fn forward_batched_impl(&self, stm: &Tensor<B, 2>, ntm: &Tensor<B, 2>, bucket_idx: usize) -> Tensor<B, 1> {
        // FT: [batch, 1260] → [batch, 1024]
        let ft_stm = self.ft.forward(stm.clone());
        let ft_ntm = self.ft.forward(ntm.clone());

        // SCReLU: clamp(0, 255)² applied element-wise — shape unchanged [batch, 1024]
        let clamped_stm = ft_stm.clamp(0.0, 255.0);
        let clamped_ntm = ft_ntm.clamp(0.0, 255.0);
        let stm_act = clamped_stm.clone() * clamped_stm;
        let ntm_act = clamped_ntm.clone() * clamped_ntm;

        // Concatenate along feature dim: [batch, 2048]
        let combined = Tensor::cat(vec![stm_act, ntm_act], 1);

        // Output projection: [batch, 2048] → [batch, 8]
        let out = self.out.forward(combined);

        // Gather bucket column for each sample in batch: [batch]
        // Use Tensor::full to broadcast single bucket_idx to all batch elements
        let batch_size = out.shape()[0];
        let bucket_idx_tensor = Tensor::full([batch_size, 1], bucket_idx as i64, &out.device());
        out.gather(1, bucket_idx_tensor).reshape([batch_size])
    }

    /// Batched forward: compute bucket-selected output for a batch of samples.
    /// [batch, 1260] × 2 → [batch] raw bucket values (before tanh).
    #[allow(unused)]
    pub fn forward_batched_with_bucket(
        &self,
        stm: &Tensor<B, 2>,
        ntm: &Tensor<B, 2>,
        bucket_idx: usize,
    ) -> Tensor<B, 1> {
        self.forward_batched_impl(stm, ntm, bucket_idx)
    }

    /// Batched forward: return all 8 bucket outputs for each sample in the batch.
    /// [batch, 1260] × 2 → [batch, 8] raw bucket values (before tanh).
    pub fn forward_batched_all_buckets(&self, stm: &Tensor<B, 2>, ntm: &Tensor<B, 2>) -> Tensor<B, 2> {
        // FT: [batch, 1260] → [batch, 1024]
        let ft_stm = self.ft.forward(stm.clone());
        let ft_ntm = self.ft.forward(ntm.clone());

        // SCReLU: clamp(0, 255)² applied element-wise — shape unchanged [batch, 1024]
        let clamped_stm = ft_stm.clamp(0.0, 255.0);
        let clamped_ntm = ft_ntm.clamp(0.0, 255.0);
        let stm_act = clamped_stm.clone() * clamped_stm;
        let ntm_act = clamped_ntm.clone() * clamped_ntm;

        // Concatenate along feature dim: [batch, 2048]
        let combined = Tensor::cat(vec![stm_act, ntm_act], 1);

        // Output projection: [batch, 2048] → [batch, 8]
        self.out.forward(combined)
    }
}

#[cfg(feature = "train")]
impl<B: Backend> Default for NNUEFeedForwardBurn<B> {
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
    /// NN raw score in centipawns. Range [-400, 400].
    pub nn_score: f32,
    /// Additive correction. Range [-400, 400] centipawns.
    pub correction: f32,
}

/// Global NN instance, lazily initialized from file if present, random otherwise.
static NN_NET: std::sync::LazyLock<NNUEFeedForward> =
    std::sync::LazyLock::new(|| NNUEFeedForward::load_from_file("nn_weights.bin"));

/// Hybrid NN + handcrafted evaluation.
///
/// Blending formula (per spec):
///   final_score = (alpha / (alpha + beta)) * nn_score
///               + (beta / (alpha + beta)) * handcrafted_score
///               + correction
///
/// Where alpha, beta ∈ [0.05, 0.95] (sigmoid-clamped), nn_score ∈ [-400, 400],
/// correction ∈ [-400, 400], handcrafted_score is in centipawns.
#[allow(dead_code)]
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);

    // Encode board into NNUE dual-perspective input planes
    let (_stm, _ntm) = NNInputPlanes::from_board(board);
    let _non_king_count = crate::nnue_input::count_non_king_pieces(board);

    let output = NN_NET.forward_output(&_stm.data, &_ntm.data, _non_king_count);

    // Fixed 75% NN / 25% handcrafted blend.
    // Both nn_score and handcrafted are normalized to [-400, 400] via tanh,
    // then blended and offset by correction.
    let nn_score = output.nn_score;
    let handcrafted_norm = (handcrafted as f32 / (MATE_SCORE as f32 / 4.0)).tanh() * 400.0;
    let correction = output.correction;

    let blended = 0.75 * nn_score + 0.25 * handcrafted_norm + correction;
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
    fn test_nnue_forward_output_ranges() {
        let net = NNUEFeedForward::random();
        let board = Board::new(RuleSet::Official, 1);
        let (stm, ntm) = NNInputPlanes::from_board(&board);
        let non_king_count = crate::nnue_input::count_non_king_pieces(&board);
        let output = net.forward_output(&stm.data, &ntm.data, non_king_count);

        // Alpha and beta should be in [0.05, 0.95] for hybrid eval
        // But NNUE returns alpha=1.0, beta=0.0 (pure NN)
        assert!(output.alpha >= 0.0 && output.alpha <= 1.0,
            "alpha {} out of range [0, 1]", output.alpha);
        assert!(output.beta >= 0.0 && output.beta <= 1.0,
            "beta {} out of range [0, 1]", output.beta);

        // NN score should be in roughly [-400, 400]
        assert!(output.nn_score >= -400.0 && output.nn_score <= 400.0,
            "nn_score {} out of range [-400, 400]", output.nn_score);
        assert!(output.correction >= -400.0 && output.correction <= 400.0,
            "correction {} out of range [-400, 400]", output.correction);
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

    #[test]
    fn test_nnue_to_from_bytes() {
        let net = NNUEFeedForward::random();
        let bytes = net.to_bytes().unwrap();
        let loaded = NNUEFeedForward::from_bytes(&bytes).unwrap();

        // Check ft_bias
        assert_eq!(net.ft_bias, loaded.ft_bias);
        // Check out_bias
        assert_eq!(net.out_bias, loaded.out_bias);
        // Check out_weights
        assert_eq!(net.out_weights, loaded.out_weights);
        // Check ft_weights
        assert_eq!(net.ft_weights.len(), loaded.ft_weights.len());
        for (a, b) in net.ft_weights.iter().zip(loaded.ft_weights.iter()) {
            assert_eq!(a.vals, b.vals);
        }
    }
}
