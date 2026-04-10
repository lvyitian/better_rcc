//! Neural network training module.
//! Behind `#[cfg(feature = "train")]`.

#[cfg(feature = "train")]
pub mod nn_train {
    use bincode;
    use serde::{Deserialize, Serialize};
    use crate::eval::{Board, Color};
    use crate::nn_eval::{InputPlanes, CompactResNetBurn, NNOutput};
    use burn::optim::AdamW;
    use burn::prelude::*;
    use burn_ndarray::NdArray;
    use burn_autodiff::Autodiff;

    /// Training backend: Autodiff wrapping NdArray for gradient computation.
    pub type TrainBackend = Autodiff<NdArray<f32>>;
    /// Inference backend: plain NdArray.
    pub type InferenceBackend = NdArray<f32>;

    /// A single training sample: flat input planes + label + side to move.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrainingSample {
        /// Flat [f32; 3420] InputPlanes data stored as Vec for serde compatibility
        pub planes: Vec<f32>,
        /// Normalized label: clamp(score/400.0, -1.0, 1.0) for supervised,
        /// or +1.0/0.0/-1.0 for game outcome
        pub label: f32,
        /// 0 = Red to move, 1 = Black to move
        pub side_to_move: u8,
    }

    impl TrainingSample {
        /// Create a TrainingSample from board + score (supervised).
        pub fn from_board(board: &Board, side_to_move: Color, score: i32) -> Self {
            let planes = InputPlanes::from_board(board, side_to_move);
            let label = (score as f32 / 400.0).clamp(-1.0, 1.0);
            let side_to_move = match side_to_move {
                Color::Red => 0,
                Color::Black => 1,
            };
            Self {
                planes: planes.into_vec(),
                label,
                side_to_move,
            }
        }

        /// Reconstruct InputPlanes from this sample.
        pub fn to_input_planes(&self) -> InputPlanes {
            InputPlanes::from_flat(self.planes.clone())
        }
    }

    /// Serialize training samples to binary file.
    /// Format: [u32 count][TrainingSample x count]
    pub fn save_training_data(samples: &[TrainingSample], path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let count = samples.len() as u32;
        writer.write_all(&count.to_le_bytes())?;
        for sample in samples {
            let encoded: Vec<u8> = bincode::serialize(sample)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
            writer.write_all(&encoded)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Deserialize training samples from binary file.
    pub fn load_training_data(path: &str) -> std::io::Result<Vec<TrainingSample>> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let mut size_buf = [0u8; 4];
            reader.read_exact(&mut size_buf)?;
            let size = u32::from_le_bytes(size_buf) as usize;
            let mut data = vec![0u8; size];
            reader.read_exact(&mut data)?;
            if let Ok(sample) = bincode::deserialize(&data) {
                samples.push(sample);
            }
        }
        Ok(samples)
    }

    /// Self-play game collector for training data.
    pub struct SelfPlayCollector {
        samples: Vec<TrainingSample>,
        max_depth: u8,
    }

    impl SelfPlayCollector {
        pub fn new(max_depth: u8) -> Self {
            Self {
                samples: Vec::new(),
                max_depth,
            }
        }

        /// Run one self-play game, collecting positions with scores.
        /// Returns game outcome: +1.0 (Red wins), 0.0 (draw), -1.0 (Black wins).
        pub fn run_game(&mut self, rule_set: crate::RuleSet, order: u8) -> f32 {
            use crate::search;
            use crate::eval::eval::handcrafted_evaluate;

            let mut board = Board::new(rule_set, order);
            let mut outcome = 0.0f32; // draw default

            loop {
                // Check game over
                if let Some(winner) = board.get_winner() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
                    break;
                }
                if board.is_repetition_violation().is_some() {
                    break;
                }

                // Collect position before move
                let side = board.current_side;
                let score = search::find_best_move(&mut board, self.max_depth, side)
                    .map(|_| {
                        // Use handcrafted eval as proxy score for supervised training
                        handcrafted_evaluate(&board, side, false) as f32
                    })
                    .unwrap_or(0.0);

                // Store sample with position before the move
                let sample = TrainingSample::from_board(&board, side, score as i32);
                self.samples.push(sample);

                // Make a move
                if let Some(action) = search::find_best_move(&mut board, self.max_depth, side) {
                    board.make_move(action);
                } else {
                    break;
                }
            }

            outcome
        }

        pub fn into_samples(self) -> Vec<TrainingSample> {
            self.samples
        }
    }

    /// Phase 1: Supervised pretraining — MSE on score head only.
    ///
    /// Uses burn autodiff backend for gradient computation.
    /// The alpha/beta heads are frozen (no gradient) during this phase.
    ///
    /// **NOTE:** This is a simplified placeholder. Full training requires
    /// proper burn autodiff integration with `require_grad()` and `backward()`
    /// on a loss tensor. The actual implementation is pending.
    pub fn train_supervised(
        _net: &mut CompactResNetBurn<TrainBackend>,
        train_data: &[TrainingSample],
        _val_data: &[TrainingSample],
        epochs: usize,
        batch_size: usize,
        _lr: f64,
    ) {
        eprintln!("Starting supervised training: {} train samples, {} epochs, batch={}",
                 train_data.len(), epochs, batch_size);
        eprintln!("NOTE: Full burn autodiff training loop pending implementation.");
        eprintln!("Self-play data collection via SelfPlayCollector::run_game() is working.");

        // TODO: Full training requires:
        // 1. Create tensors with .require_grad()
        // 2. Forward pass through the network
        // 3. Compute MSE loss on score head
        // 4. Call loss.backward() for gradients
        // 5. Use AdamW optimizer to update weights
        // For now, this is a placeholder that validates the data pipeline.

        for epoch in 0..epochs {
            let num_batches = (train_data.len() + batch_size - 1) / batch_size;
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(train_data.len());
                eprintln!("  Epoch {} batch {}/{}: {}-{} samples",
                         epoch, batch_idx, num_batches, start, end);
            }
        }
    }

    /// Compute validation loss on a trained network.
    /// Returns MSE between predicted nn_score and labels.
    pub fn compute_val_loss(net: &mut CompactResNetBurn<TrainBackend>, val_data: &[TrainingSample]) -> f32 {
        use burn::tensor::Tensor;
        let device = <TrainBackend as Backend>::Device::default();
        let mut total_loss = 0.0f32;

        for sample in val_data {
            let arr: [f32; 3420] = {
                let mut a = [0.0f32; 3420];
                a.copy_from_slice(&sample.planes);
                a
            };
            let flat: Tensor<TrainBackend, 1> = Tensor::from_data(TensorData::from(arr), &device);
            let input: Tensor<TrainBackend, 4> = flat.reshape([1, 38, 9, 10]);

            // Forward pass through the network
            let x = net.init_conv.forward(input);
            let x = net.relu.forward(x);
            let x = net.b1.forward(x);
            let x = net.b2.forward(x);
            let x = net.b3.forward(x);
            let x = net.b4.forward(x);
            let x = net.b5.forward(x);
            let x = net.b6.forward(x);
            let pooled = x.mean_dim(3).mean_dim(2).reshape([1, 64]);
            let dense_out = net.relu.forward(net.dense.forward(pooled));
            let score_out = net.score_head.forward(dense_out);

            // Extract scalar from [1, 1] tensor
            let score_raw: f32 = score_out.to_data().as_slice().expect("expected 1x1")[0];
            let nn_score = score_raw.tanh() * 300.0;

            let loss = (nn_score / 300.0 - sample.label).powi(2);
            total_loss += loss;
        }
        total_loss / val_data.len() as f32
    }
}
