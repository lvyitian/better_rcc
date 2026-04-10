// Neural network evaluation module
// Converts board state to input planes for NN inference

use crate::{Board, Color, PieceType};
use crate::eval::eval::{handcrafted_evaluate, game_phase};

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

#[allow(dead_code)]
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    // For now, just call the handcrafted evaluation
    // TODO: Replace with NN inference when model is loaded
    handcrafted_evaluate(board, side, initiative)
}
