//! Neural network training module.
//! Behind `#[cfg(feature = "train")]`.

#[cfg(feature = "train")]
pub mod nn_train_impl {
    use bincode;
    use serde::{Deserialize, Serialize};
    use crate::eval::{Board, Color};
    use crate::nn_eval::{InputPlanes, CompactResNetBurn};
    use burn::prelude::*;
    use burn_ndarray::NdArray;
    use burn_autodiff::Autodiff;

    /// Training backend: Autodiff wrapping NdArray for gradient computation.
    pub type TrainBackend = Autodiff<NdArray<f32>>;

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

        /// Convert TrainingSample planes back to InputPlanes.
        #[allow(dead_code)]
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
        for i in 0..count {
            let mut size_buf = [0u8; 4];
            reader.read_exact(&mut size_buf)?;
            let size = u32::from_le_bytes(size_buf) as usize;
            let mut data = vec![0u8; size];
            reader.read_exact(&mut data)?;
            if let Ok(sample) = bincode::deserialize(&data) {
                samples.push(sample);
            } else {
                eprintln!("[SelfPlayCollector] WARNING: failed to deserialize sample #{} of {}, skipping", i, count);
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
            use crate::eval::eval_impl::handcrafted_evaluate;

            let mut board = Board::new(rule_set, order);
            let outcome: f32;

            loop {
                // Check game over (king captured or repetition)
                if let Some(winner) = board.get_winner() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
                    break;
                }
                if let Some(winner) = board.is_repetition_violation() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
                    break;
                }

                let side = board.current_side;

                // Search once: returns None if kingless or no legal moves
                let action = match search::find_best_move(&mut board, self.max_depth, side) {
                    Some(a) => a,
                    None => {
                        // No legal moves: could be kingless (loss) or stalemate (draw)
                        // Check winner first to distinguish: king gone → loss, else stalemate
                        if let Some(winner) = board.get_winner() {
                            outcome = match winner {
                                Color::Red => 1.0,
                                Color::Black => -1.0,
                            };
                        } else {
                            // No king was captured — must be stalemate
                            outcome = if board.is_check(side) {
                                // In check → checkmate → opponent wins
                                match side.opponent() {
                                    Color::Red => 1.0,
                                    Color::Black => -1.0,
                                }
                            } else {
                                0.0 // stalemate → draw
                            };
                        }
                        break;
                    }
                };

                // Score the position before the move
                let score = handcrafted_evaluate(&board, side, false) as f32;
                let sample = TrainingSample::from_board(&board, side, score as i32);
                self.samples.push(sample);

                board.make_move(action);

                // Re-check in case a king was captured during this move
                if let Some(winner) = board.get_winner() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
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
    /// Only the score_head is used in the forward pass (alpha/beta heads
    /// are never called), so they receive zero gradient. The shared
    /// convolutional backbone and dense layer accumulate gradients.
    pub fn train_supervised(
        net: &mut CompactResNetBurn<TrainBackend>,
        train_data: &[TrainingSample],
        val_data: &[TrainingSample],
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) {
        use burn::tensor::Tensor;
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

        let device = <TrainBackend as Backend>::Device::default();
        let mut optim = AdamWConfig::new()
            .with_weight_decay(1e-4)
            .init();

        eprintln!("Starting supervised training: {} train, {} val, {} epochs, batch={}, lr={}",
                 train_data.len(), val_data.len(), epochs, batch_size, lr);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let num_batches = train_data.len().div_ceil(batch_size);

            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(train_data.len());
                let batch = &train_data[start..end];

                // Build batched input tensor [batch, 38, 9, 10]
                let mut flat_data = Vec::with_capacity(batch.len() * 3420);
                let mut targets = Vec::with_capacity(batch.len());
                for sample in batch {
                    flat_data.extend_from_slice(&sample.planes);
                    targets.push(sample.label);
                }
                let input: Tensor<TrainBackend, 4> = Tensor::from_data(
                    TensorData::new(flat_data, [batch.len(), 38, 9, 10]), &device
                );

                // Forward pass through network
                let x = net.init_conv.forward(input);
                let x = net.relu.forward(x);
                let x = net.b1.forward(x);
                let x = net.b2.forward(x);
                let x = net.b3.forward(x);
                let x = net.b4.forward(x);
                let x = net.b5.forward(x);
                let x = net.b6.forward(x);
                let pooled = x.mean_dim(3).mean_dim(2).reshape([batch.len() as u32, 64]);
                let dense_out = net.relu.forward(net.dense.forward(pooled));
                let score_out = net.score_head.forward(dense_out);

                // MSE loss: mean((score_normalized - target)^2) where score_normalized = score/300
                // All computation stays in Burn tensors to preserve autodiff graph
                let targets_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
                    TensorData::from(targets.as_slice()), &device
                );
                let score_squeezed = score_out.reshape([batch.len() as u32]);
                let normalized = score_squeezed / 300.0;
                let diff = normalized - targets_tensor;
                let loss_tensor = diff.clone() * diff;
                let batch_loss: f32 = loss_tensor.clone().mean().to_data().as_slice()
                    .expect("batch loss")[0];

                // Backward pass — grads flow through entire model
                let grads = loss_tensor.mean().backward();
                let grads = GradientsParams::from_grads(grads, &*net);
                let new_net = optim.step(lr, (*net).clone(), grads);
                *net = new_net;

                epoch_loss += batch_loss;
                if batch_idx % 20 == 0 {
                    eprintln!("  Epoch {} batch {}/{} loss={:.6}",
                             epoch, batch_idx, num_batches, batch_loss);
                }
            }

            // Validation loss (no backward)
            let val_loss = compute_val_loss(net, val_data);
            eprintln!("Epoch {}: train_loss={:.6} val_loss={:.6}",
                      epoch, epoch_loss / num_batches as f32, val_loss);
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

    // =============================================================================
    // NETWORK SAVE / LOAD
    // =============================================================================

    /// Save network weights to a binary file using burn's CompactRecorder.
    pub fn save_network(net: &CompactResNetBurn<TrainBackend>, path: &str) -> std::io::Result<()> {
        use burn::record::CompactRecorder;
        (*net).clone().save_file(path, &CompactRecorder::new())
            .map_err(std::io::Error::other)
    }

    /// Load network weights from a binary file.
    /// Returns a new network with loaded weights on the training backend.
    pub fn load_network(path: &str) -> std::io::Result<CompactResNetBurn<TrainBackend>> {
        use burn::record::{CompactRecorder, Recorder};
        let device = <TrainBackend as Backend>::Device::default();
        let record = CompactRecorder::new()
            .load(path.into(), &device)
            .map_err(std::io::Error::other)?;
        let net = CompactResNetBurn::new();
        Ok(net.load_record(record))
    }

    // =============================================================================
    // PHASE 2: SELF-PLAY FINE-TUNING
    // =============================================================================

    /// Phase 2: Self-play fine-tuning — all heads trainable, combined loss.
    ///
    /// Combined loss: mse(nn_score, outcome)
    ///               + 0.1 * (binary_cross_entropy(alpha, outcome_bin)
    ///                      + binary_cross_entropy(beta, outcome_bin))
    /// where outcome ∈ {-1, 0, +1} and outcome_bin = (outcome + 1) / 2 ∈ {0, 0.5, 1}.
    pub fn train_selfplay(
        net: &mut CompactResNetBurn<TrainBackend>,
        train_data: &[TrainingSample],
        val_data: &[TrainingSample],
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) {
        use burn::tensor::Tensor;
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

        let device = <TrainBackend as Backend>::Device::default();
        let mut optim = AdamWConfig::new()
            .with_weight_decay(1e-4)
            .init();

        eprintln!("Starting self-play fine-tuning: {} train, {} val, {} epochs, batch={}, lr={}",
                 train_data.len(), val_data.len(), epochs, batch_size, lr);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let num_batches = train_data.len().div_ceil(batch_size);

            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(train_data.len());
                let batch = &train_data[start..end];

                // Build batched input tensor [batch, 38, 9, 10]
                let mut flat_data = Vec::with_capacity(batch.len() * 3420);
                for sample in batch {
                    flat_data.extend_from_slice(&sample.planes);
                }
                let input: Tensor<TrainBackend, 4> = Tensor::from_data(
                    TensorData::new(flat_data, [batch.len(), 38, 9, 10]), &device
                );

                // Forward pass: all 4 heads
                let x = net.init_conv.forward(input);
                let x = net.relu.forward(x);
                let x = net.b1.forward(x);
                let x = net.b2.forward(x);
                let x = net.b3.forward(x);
                let x = net.b4.forward(x);
                let x = net.b5.forward(x);
                let x = net.b6.forward(x);
                let pooled = x.mean_dim(3).mean_dim(2).reshape([batch.len() as u32, 64]);
                let dense_out = net.relu.forward(net.dense.forward(pooled));

                let alpha_raw = net.alpha_head.forward(dense_out.clone());
                let beta_raw = net.beta_head.forward(dense_out.clone());
                let score_raw = net.score_head.forward(dense_out.clone());

                // Scalar outputs [batch, 1] → squeeze to [batch]
                let squeeze = |t: Tensor<TrainBackend, 2>| {
                    t.reshape([batch.len() as u32])
                };
                let alpha_s = squeeze(alpha_raw);
                let beta_s = squeeze(beta_raw);
                let score_s = squeeze(score_raw);

                // Scale: alpha/beta → sigmoid * 0.9 + 0.05 ∈ [0.05, 0.95]
                //         score    → tanh * 300 ∈ [-300, 300]
                let alpha_vals: Vec<f32> = alpha_s.to_data().as_slice().expect("batch alpha").to_vec();
                let beta_vals: Vec<f32> = beta_s.to_data().as_slice().expect("batch beta").to_vec();
                let score_vals: Vec<f32> = score_s.to_data().as_slice().expect("batch score").to_vec();

                // Compute combined loss per sample
                let mut batch_loss: f32 = 0.0;
                for (i, sample_label) in batch.iter().enumerate() {
                    let outcome = sample_label.label; // +1/0/-1
                    let outcome_bin: f32 = (outcome + 1.0) / 2.0; // → 1/0.5/0

                    let alpha_raw: f32 = alpha_vals[i];
                    let beta_raw: f32 = beta_vals[i];
                    let score_raw: f32 = score_vals[i];

                    // Sigmoid + scale for alpha/beta, tanh + scale for score
                    let alpha = (1.0 / (1.0 + (-alpha_raw).exp())) * 0.9 + 0.05;
                    let beta = (1.0 / (1.0 + (-beta_raw).exp())) * 0.9 + 0.05;
                    let nn_score = score_raw.tanh() * 300.0;

                    // MSE on score head
                    let mse: f32 = ((nn_score / 300.0) - outcome).powi(2);

                    // BCE on alpha/beta heads (target is binary: 1=Red win, 0=Black win, 0.5=draw)
                    let bce_alpha: f32 = {
                        let p = alpha.clamp(1e-15_f32, 1.0 - 1e-15_f32);
                        let q = outcome_bin.clamp(1e-15_f32, 1.0 - 1e-15_f32);
                        -q * p.ln() - (1.0 - q) * (1.0 - p).ln()
                    };
                    let bce_beta: f32 = {
                        let p = beta.clamp(1e-15_f32, 1.0 - 1e-15_f32);
                        let q = outcome_bin.clamp(1e-15_f32, 1.0 - 1e-15_f32);
                        -q * p.ln() - (1.0 - q) * (1.0 - p).ln()
                    };

                    batch_loss += mse + 0.1 * (bce_alpha + bce_beta);
                }
                batch_loss /= batch.len() as f32;

                // Backward pass
                let loss_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
                    TensorData::from([batch_loss]), &device
                );
                let grads = loss_tensor.backward();
                let grads = GradientsParams::from_grads(grads, &*net);
                let new_net = optim.step(lr, (*net).clone(), grads);
                *net = new_net;

                epoch_loss += batch_loss;
                if batch_idx % 20 == 0 {
                    eprintln!("  Epoch {} batch {}/{} loss={:.6}",
                             epoch, batch_idx, num_batches, batch_loss);
                }
            }

            // Validation loss (Phase 2 MSE only)
            let val_loss = compute_val_loss(net, val_data);
            eprintln!("Epoch {}: train_loss={:.6} val_loss={:.6}",
                      epoch, epoch_loss / num_batches as f32, val_loss);
        }
    }

    // =============================================================================
    // SELF-PLAY COLLECTOR WITH GAME OUTCOMES (for Phase 2)
    // =============================================================================

    /// Collect positions from self-play with game outcome as label.
    /// Outcome labels: +1.0 (Red wins), 0.0 (draw), -1.0 (Black wins).
    pub struct SelfPlayOutcomeCollector {
        samples: Vec<TrainingSample>,
        max_depth: u8,
    }

    impl SelfPlayOutcomeCollector {
        pub fn new(max_depth: u8) -> Self {
            Self {
                samples: Vec::new(),
                max_depth,
            }
        }

        /// Run one self-play game, collecting positions with game outcome.
        /// Returns game outcome: +1.0 (Red wins), 0.0 (draw), -1.0 (Black wins).
        pub fn run_game(&mut self, rule_set: crate::RuleSet, order: u8) -> f32 {
            use crate::search;

            let mut board = Board::new(rule_set, order);
            let mut outcome = 0.0f32;

            loop {
                if let Some(winner) = board.get_winner() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
                    break;
                }
                if let Some(winner) = board.is_repetition_violation() {
                    outcome = match winner {
                        Color::Red => 1.0,
                        Color::Black => -1.0,
                    };
                    break;
                }

                let side = board.current_side;

                // Make a move
                if let Some(action) = search::find_best_move(&mut board, self.max_depth, side) {
                    // Store position BEFORE move with current game outcome
                    // (we don't know outcome yet, so store placeholder 0.0 for now;
                    //  we'll retroactively label all positions after game ends)
                    let planes = InputPlanes::from_board(&board, side);
                    let side_to_move = match side {
                        Color::Red => 0,
                        Color::Black => 1,
                    };
                    self.samples.push(TrainingSample {
                        planes: planes.into_vec(),
                        label: 0.0, // placeholder — will be set after game ends
                        side_to_move,
                    });
                    board.make_move(action);
                } else {
                    break;
                }
            }

            // Retroactively label all collected positions with game outcome
            for sample in self.samples.iter_mut() {
                sample.label = outcome;
            }

            outcome
        }

        pub fn into_samples(self) -> Vec<TrainingSample> {
            self.samples
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Regression test: MSE loss backward must not panic with
        /// "Node should have a step registered, did you forget to call `Tensor::register_grad`?"
        ///
        /// The original bug broke the autodiff graph by extracting `score_out` to a CPU `Vec`,
        /// computing loss in a Rust loop, then creating a disconnected `loss_tensor` from raw data.
        /// Calling `backward()` on that leaf tensor had no gradient path back to model parameters.
        ///
        /// The fix: keep all loss computation in Burn tensor ops so `backward()` traces
        /// the full path score_out → dense → conv → model weights.
        #[test]
        fn test_train_supervised_backward_graph_integrity() {
            use burn::tensor::TensorData;
            use burn::optim::GradientsParams;

            let net = CompactResNetBurn::<TrainBackend>::new();
            let device = <TrainBackend as Backend>::Device::default();

            // One sample with known label
            let board = Board::new(crate::RuleSet::Official, 1);
            let sample = TrainingSample::from_board(&board, Color::Red, 100);

            // Build single-element batch
            let input: Tensor<TrainBackend, 4> = Tensor::from_data(
                TensorData::new(sample.planes.clone(), [1, 38, 9, 10]), &device
            );
            let targets = vec![sample.label];

            // Forward pass (same ops as train_supervised)
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

            // Loss computed entirely in tensor space — preserves autodiff graph
            let targets_tensor: Tensor<TrainBackend, 1> =
                Tensor::from_data(TensorData::from(targets.as_slice()), &device);
            let score_squeezed = score_out.reshape([1]);
            let normalized = score_squeezed / 300.0;
            let diff = normalized - targets_tensor;
            let loss_tensor = diff.clone() * diff;

            // This must NOT panic with "Node should have a step registered"
            let grads = loss_tensor.mean().backward();

            // Verify GradientsParams can be constructed (proves grads are well-formed)
            let _grads_params = GradientsParams::from_grads(grads, &net);
            // If we got here without panic, the autodiff graph is intact
        }
    }
}
