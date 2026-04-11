//! Neural network training module.
//! Behind `#[cfg(feature = "train")]`.

#[cfg(feature = "train")]
pub mod nn_train_impl {
    use bincode;
    use serde::{Deserialize, Serialize};
    use crate::eval::{Board, Color};
    use crate::nn_eval::{InputPlanes, NNUEFeedForwardBurn};
    use burn::prelude::*;
    use burn_ndarray::NdArray;
    use burn_autodiff::Autodiff;

    /// Training backend: Autodiff wrapping NdArray for gradient computation.
    pub type TrainBackend = Autodiff<NdArray<f32>>;

    /// A single training sample: flat input planes + label + side to move.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrainingSample {
        /// Flat [f32; 1260] NNInputPlanes data (stm perspective) stored as Vec for serde compatibility
        pub stm_planes: Vec<f32>,
        /// Flat [f32; 1260] NNInputPlanes data (ntm perspective) stored as Vec
        pub ntm_planes: Vec<f32>,
        /// Non-king piece count used to select bucket (0–30)
        pub non_king_count: u8,
        /// Normalized label: clamp(score/400.0, -1.0, 1.0) for supervised,
        /// or +1.0/0.0/-1.0 for game outcome
        pub label: f32,
        /// 0 = Red to move, 1 = Black to move
        pub side_to_move: u8,
    }

    impl TrainingSample {
        /// Create a TrainingSample from board + score (supervised).
        pub fn from_board(board: &Board, side_to_move: Color, score: i32) -> Self {
            use crate::nnue_input::{NNInputPlanes, count_non_king_pieces};
            let (stm, ntm) = NNInputPlanes::from_board(board);
            let label = (score as f32 / 400.0).clamp(-1.0, 1.0);
            let side_to_move = match side_to_move {
                Color::Red => 0,
                Color::Black => 1,
            };
            let non_king_count = count_non_king_pieces(board);
            Self {
                stm_planes: stm.data.to_vec(),
                ntm_planes: ntm.data.to_vec(),
                non_king_count,
                label,
                side_to_move,
            }
        }

        /// Convert TrainingSample planes back to InputPlanes (legacy).
        #[allow(dead_code)]
        pub fn to_input_planes(&self) -> InputPlanes {
            InputPlanes::from_flat(self.stm_planes.clone())
        }
    }

    /// Serialize training samples to a zstd-compressed binary file.
    /// Format: [u32 uncompressed_size][u32 count][TrainingSample x count] (all zstd-compressed)
    pub fn save_training_data(samples: &[TrainingSample], path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Pre-encode all samples to get uncompressed size
        let mut all_data: Vec<u8> = Vec::new();
        let count = samples.len() as u32;
        all_data.extend_from_slice(&count.to_le_bytes());
        for sample in samples {
            let encoded: Vec<u8> = bincode::serialize(sample)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            all_data.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            all_data.extend_from_slice(&encoded);
        }

        // Compress with zstd
        let compressed = zstd::encode_all(all_data.as_slice(), 3)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Write uncompressed size header + compressed data
        let uncompressed_size = all_data.len() as u32;
        writer.write_all(&uncompressed_size.to_le_bytes())?;
        writer.write_all(&compressed)?;
        writer.flush()?;
        Ok(())
    }

    /// Deserialize training samples from a zstd-compressed binary file.
    pub fn load_training_data(path: &str) -> std::io::Result<Vec<TrainingSample>> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header: uncompressed size + compressed data
        let mut size_buf = [0u8; 4];
        reader.read_exact(&mut size_buf)?;
        let uncompressed_size = u32::from_le_bytes(size_buf) as usize;

        // Read rest as compressed data
        let mut compressed_data = Vec::new();
        reader.read_to_end(&mut compressed_data)?;

        // Decompress
        let all_data = zstd::decode_all(compressed_data.as_slice())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if all_data.len() != uncompressed_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("decompressed size mismatch: expected {}, got {}", uncompressed_size, all_data.len())
            ));
        }

        // Parse: [u32 count][samples...]
        let mut offset = 0;
        let count = u32::from_le_bytes(all_data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;

        let mut samples = Vec::with_capacity(count);
        for i in 0..count {
            let size = u32::from_le_bytes(all_data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            let data = &all_data[offset..offset+size];
            offset += size;
            if let Ok(sample) = bincode::deserialize(data) {
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
        net: &mut NNUEFeedForwardBurn<TrainBackend>,
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

                // Build batched input tensors for NNUE: [batch, 1260] each for stm and ntm
                let mut stm_flat = Vec::with_capacity(batch.len() * 1260);
                let mut ntm_flat = Vec::with_capacity(batch.len() * 1260);
                let mut targets = Vec::with_capacity(batch.len());
                let bucket_idx = crate::nnue_input::bucket_index(batch.iter().map(|s| s.non_king_count).max().unwrap_or(0)) as i64;
                for sample in batch {
                    stm_flat.extend_from_slice(&sample.stm_planes);
                    ntm_flat.extend_from_slice(&sample.ntm_planes);
                    targets.push(sample.label);
                }
                let stm_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
                    TensorData::new(stm_flat, [batch.len(), 1260]), &device
                );
                let ntm_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
                    TensorData::new(ntm_flat, [batch.len(), 1260]), &device
                );

                // Forward pass: NNUEFeedForwardBurn → gather bucket-selected outputs → tanh * 400
                let score_raw: Tensor<TrainBackend, 1> = net.forward_batched_with_bucket(&stm_tensor, &ntm_tensor, bucket_idx as usize);

                // Scale to [-400, 400] via tanh, normalize to [-1, 1]
                let nn_score = score_raw.tanh() * 400.0;
                let normalized = nn_score / 400.0;

                // MSE loss: mean((nn_score_normalized - target)^2)
                let targets_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
                    TensorData::from(targets.as_slice()), &device
                );
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
    pub fn compute_val_loss(net: &mut NNUEFeedForwardBurn<TrainBackend>, val_data: &[TrainingSample]) -> f32 {
        use burn::tensor::Tensor;
        let device = <TrainBackend as Backend>::Device::default();
        let mut total_loss = 0.0f32;

        for sample in val_data {
            let stm_arr: [f32; 1260] = {
                let mut a = [0.0f32; 1260];
                a.copy_from_slice(&sample.stm_planes);
                a
            };
            let ntm_arr: [f32; 1260] = {
                let mut a = [0.0f32; 1260];
                a.copy_from_slice(&sample.ntm_planes);
                a
            };
            let stm_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(TensorData::from(stm_arr), &device);
            let ntm_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(TensorData::from(ntm_arr), &device);

            // Forward pass: get bucket-selected output, apply tanh * 400
            let bucket_idx = crate::nnue_input::bucket_index(sample.non_king_count);
            let bucket_tensor = net.forward_with_bucket(&stm_tensor, &ntm_tensor, bucket_idx);
            let score_raw: f32 = bucket_tensor.to_data().as_slice().expect("expected 1x1")[0];
            // Normalize to [-1, 1] to match label units (same as training)
            let nn_score = score_raw.tanh();
            let normalized = nn_score - sample.label;
            let loss = normalized * normalized;
            total_loss += loss;
        }
        total_loss / val_data.len() as f32
    }

    // =============================================================================
    // NETWORK SAVE / LOAD
    // =============================================================================

    /// Save network weights to a binary file using burn's CompactRecorder.
    pub fn save_network(net: &NNUEFeedForwardBurn<TrainBackend>, path: &str) -> std::io::Result<()> {
        use burn::record::CompactRecorder;
        (*net).clone().save_file(path, &CompactRecorder::new())
            .map_err(std::io::Error::other)
    }

    /// Load network weights from a binary file.
    /// Returns a new network with loaded weights on the training backend.
    pub fn load_network(path: &str) -> std::io::Result<NNUEFeedForwardBurn<TrainBackend>> {
        use burn::record::{CompactRecorder, Recorder};
        let device = <TrainBackend as Backend>::Device::default();
        let record = CompactRecorder::new()
            .load(path.into(), &device)
            .map_err(std::io::Error::other)?;
        let net = NNUEFeedForwardBurn::new();
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
        net: &mut NNUEFeedForwardBurn<TrainBackend>,
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

                // Build batched input tensors for NNUE: [batch, 1260] each for stm and ntm
                let mut stm_flat = Vec::with_capacity(batch.len() * 1260);
                let mut ntm_flat = Vec::with_capacity(batch.len() * 1260);
                let mut outcomes = Vec::with_capacity(batch.len());
                for sample in batch {
                    stm_flat.extend_from_slice(&sample.stm_planes);
                    ntm_flat.extend_from_slice(&sample.ntm_planes);
                    outcomes.push(sample.label);
                }
                let stm_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
                    TensorData::new(stm_flat, [batch.len(), 1260]), &device
                );
                let ntm_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
                    TensorData::new(ntm_flat, [batch.len(), 1260]), &device
                );

                // Forward pass: NNUEFeedForwardBurn → get all 8 bucket outputs per sample
                let all_outputs: Tensor<TrainBackend, 2> = net.forward_batched_all_buckets(&stm_tensor, &ntm_tensor);

                // Gather bucket-selected outputs using per-sample bucket indices
                let bucket_indices: Vec<i64> = batch.iter()
                    .map(|s| crate::nnue_input::bucket_index(s.non_king_count) as i64)
                    .collect();
                let bucket_idx_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
                    TensorData::from(bucket_indices.as_slice()), &device
                );
                let score_raw: Tensor<TrainBackend, 1> = all_outputs.gather(1, bucket_idx_tensor.reshape([batch.len() as u32, 1]).int()).reshape([batch.len() as u32]);

                // Scale to [-400, 400] via tanh, normalize to [-1, 1]
                let nn_score = score_raw.tanh() * 400.0;

                // MSE loss on score: mean((nn_score/400 - outcome)^2)
                // nn_score/400 ∈ [-1, 1] matches outcome ∈ [-1, 1]
                let outcomes_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
                    TensorData::from(outcomes.as_slice()), &device
                );
                let diff = nn_score / 400.0 - outcomes_tensor;
                let mse = diff.clone() * diff;
                let loss_tensor: Tensor<TrainBackend, 1> = mse.mean();

                // Backward pass — gradients flow through full model
                let grads = loss_tensor.backward();

                // Verify grads are well-formed by constructing GradientsParams.
                // A graph fracture (e.g., loss computed in Rust scalars) would panic here.
                let grads = GradientsParams::from_grads(grads, &*net);
                let new_net = optim.step(lr, (*net).clone(), grads);
                *net = new_net;

                // Log loss (recompute from tensor for accurate reporting)
                let batch_loss: f32 = loss_tensor.clone().mean().to_data().as_slice()
                    .expect("batch loss")[0];
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
                    use crate::nnue_input::{NNInputPlanes, count_non_king_pieces};
                    let (stm, ntm) = NNInputPlanes::from_board(&board);
                    let side_to_move = match side {
                        Color::Red => 0,
                        Color::Black => 1,
                    };
                    self.samples.push(TrainingSample {
                        stm_planes: stm.data.to_vec(),
                        ntm_planes: ntm.data.to_vec(),
                        non_king_count: count_non_king_pieces(&board),
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
        /// Verifies that the NNUEFeedForwardBurn forward pass builds a valid autodiff graph.
        /// Loss computed entirely in tensor ops; backward pass should not panic.
        #[test]
        fn test_nnue_forward_backward_graph_integrity() {
            use burn::tensor::TensorData;
            use burn::optim::GradientsParams;

            let net = NNUEFeedForwardBurn::<TrainBackend>::new();
            let device = <TrainBackend as Backend>::Device::default();

            // One sample with known label
            let board = Board::new(crate::RuleSet::Official, 1);
            let sample = TrainingSample::from_board(&board, Color::Red, 100);

            // Build [1, 1260] stm/ntm tensors
            let stm_arr: [f32; 1260] = {
                let mut a = [0.0f32; 1260];
                a.copy_from_slice(&sample.stm_planes);
                a
            };
            let ntm_arr: [f32; 1260] = {
                let mut a = [0.0f32; 1260];
                a.copy_from_slice(&sample.ntm_planes);
                a
            };
            let stm_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(TensorData::from(stm_arr), &device);
            let ntm_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(TensorData::from(ntm_arr), &device);

            // Forward pass through NNUEFeedForwardBurn
            let bucket_idx = crate::nnue_input::bucket_index(sample.non_king_count);
            let bucket_tensor = net.forward_with_bucket(&stm_tensor, &ntm_tensor, bucket_idx);

            // Loss computed entirely in tensor space — preserves autodiff graph
            // forward_with_bucket returns [1] tensor
            let score_raw = bucket_tensor.reshape([1]);
            let nn_score = score_raw.tanh() * 400.0;
            let normalized = nn_score / 400.0;
            let target_tensor: Tensor<TrainBackend, 1> =
                Tensor::from_data(TensorData::new(vec![sample.label], [1]), &device);
            let diff = normalized - target_tensor;
            let loss_tensor = diff.clone() * diff;

            // This must NOT panic with "Node should have a step registered"
            let grads = loss_tensor.backward();

            // Verify GradientsParams can be constructed (proves grads are well-formed)
            let _grads_params = GradientsParams::from_grads(grads, &net);
            // If we got here without panic, the autodiff graph is intact
        }
    }
}
