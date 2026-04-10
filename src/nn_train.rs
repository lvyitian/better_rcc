//! Neural network training module.
//! Behind `#[cfg(feature = "train")]`.

#[cfg(feature = "train")]
pub mod nn_train {
    use bincode;
    use serde::{Deserialize, Serialize};
    use crate::eval::{Board, Color};
    use crate::nn_eval::InputPlanes;

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
}
