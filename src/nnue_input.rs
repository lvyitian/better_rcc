//! NNUE (Efficiently Updatable Neural Networks) input encoding for Xiangqi.
//!
//! This module provides the input feature planes for the NNUE neural network evaluation.
//! The encoding uses a 1260-dimensional input vector organized as:
//! - Base 0-629: Our pieces (first 630 features)
//! - Base 630-1259: Their pieces (second 630 features)
//!
//! Each 630-feature block is organized as 7 piece types × 90 squares.

use crate::{Board, Coord, PieceType};

/// Input dimension: 1260 = 2 × 630 = 2 × (7 × 90)
pub const INPUT_DIM: usize = 1260;
/// Feature transform dimension: 1024 (hidden layer size of the NNUE linear transform)
pub const FT_DIM: usize = 1024;
/// Number of buckets for the value head
pub const NUM_BUCKETS: usize = 8;
/// Quantization parameter A (scale numerator)
pub const QA: i32 = 255;
/// Quantization parameter B (scale denominator base)
pub const QB: i32 = 64;
/// Score scale for NNUE output
pub const SCALE: i32 = 400;

/// Compute the bucket index for non-king piece count.
/// Bucket is used in the NNUE value head to bucketize positions by complexity.
/// Each bucket covers 4 non-king pieces (starting from 2), max bucket is 7.
#[inline]
pub fn bucket_index(non_king_count: u8) -> usize {
    (non_king_count.saturating_sub(2) / 4).min(7) as usize
}

/// NNInputPlanes wraps the 1260-dimensional input feature vector for the NNUE network.
#[derive(Debug, Clone, PartialEq)]
pub struct NNInputPlanes {
    /// The flattened feature planes: [f32; 1260]
    pub data: [f32; INPUT_DIM],
}

impl NNInputPlanes {
    /// Create a new zero-initialized NNInputPlanes.
    pub fn new() -> Self {
        Self { data: [0.0; INPUT_DIM] }
    }

    /// Encode a board position into dual-perspective NNUE input planes.
    ///
    /// Returns (stm, ntm) where:
    /// - stm (side-to-move): our pieces at base 0, their pieces at base 630
    /// - ntm (not-side-to-move): their pieces at base 0, our pieces at base 630
    ///   with vertical flip applied (rank y → 9 - y)
    pub fn from_board(board: &Board) -> (Self, Self) {
        let mut stm = Self::new();
        let mut ntm = Self::new();

        let our_color = board.current_side;
        let their_color = our_color.opponent();

        // Encode each cell for stm (side-to-move perspective)
        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.get(Coord::new(x as i8, y as i8)) {
                    let base = if piece.color == our_color { 0 } else { 630 };
                    let piece_idx = piece.piece_type as usize;
                    let square_idx = y * 9 + x;
                    let feature_idx = base + piece_idx * 90 + square_idx;
                    stm.data[feature_idx] = 1.0;
                }
            }
        }

        // Encode each cell for ntm (not-side-to-move perspective, vertically flipped)
        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.get(Coord::new(x as i8, y as i8)) {
                    // For ntm, their pieces are at base 0, our pieces at base 630
                    // But the board is vertically flipped (y → 9 - y)
                    let base = if piece.color == their_color { 0 } else { 630 };
                    let piece_idx = piece.piece_type as usize;
                    let ntm_sq = (9 - y) * 9 + x;
                    let feature_idx = base + piece_idx * 90 + ntm_sq;
                    ntm.data[feature_idx] = 1.0;
                }
            }
        }

        (stm, ntm)
    }
}

impl Default for NNInputPlanes {
    fn default() -> Self {
        Self::new()
    }
}

/// Count the total number of non-king pieces on the board.
/// This is used to determine the bucket index for the NNUE value head.
#[inline(always)]
pub fn count_non_king_pieces(board: &Board) -> u8 {
    board.bitboards.count_non_king_pieces()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Color, Piece, PieceType, RuleSet};

    #[test]
    fn test_bucket_index() {
        // Formula: (count.saturating_sub(2) / 4).min(7)
        // count=0-3 → (0-2)/4 = 0 (saturating_sub gives 0 when subtract would underflow)
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(1), 0);
        assert_eq!(bucket_index(2), 0);
        assert_eq!(bucket_index(3), 0);
        // count=4-7 → (count-2)/4
        assert_eq!(bucket_index(4), 0); // (4-2)/4 = 0
        assert_eq!(bucket_index(5), 0); // (5-2)/4 = 0
        assert_eq!(bucket_index(6), 1); // (6-2)/4 = 1
        assert_eq!(bucket_index(7), 1); // (7-2)/4 = 1
        // count=8-11
        assert_eq!(bucket_index(8), 1);  // (8-2)/4 = 1
        assert_eq!(bucket_index(9), 1);  // (9-2)/4 = 1
        assert_eq!(bucket_index(10), 2); // (10-2)/4 = 2
        assert_eq!(bucket_index(11), 2); // (11-2)/4 = 2
        // count=12-15
        assert_eq!(bucket_index(12), 2); // (12-2)/4 = 2
        assert_eq!(bucket_index(13), 2); // (13-2)/4 = 2
        assert_eq!(bucket_index(14), 3); // (14-2)/4 = 3
        assert_eq!(bucket_index(15), 3); // (15-2)/4 = 3
        // count=16-19
        assert_eq!(bucket_index(16), 3); // (16-2)/4 = 3
        assert_eq!(bucket_index(17), 3); // (17-2)/4 = 3
        assert_eq!(bucket_index(18), 4); // (18-2)/4 = 4
        assert_eq!(bucket_index(19), 4); // (19-2)/4 = 4
        // count=20-23
        assert_eq!(bucket_index(20), 4); // (20-2)/4 = 4
        assert_eq!(bucket_index(21), 4); // (21-2)/4 = 4
        assert_eq!(bucket_index(22), 5); // (22-2)/4 = 5
        assert_eq!(bucket_index(23), 5); // (23-2)/4 = 5
        // count=24-27
        assert_eq!(bucket_index(24), 5); // (24-2)/4 = 5
        assert_eq!(bucket_index(25), 5); // (25-2)/4 = 5
        assert_eq!(bucket_index(26), 6); // (26-2)/4 = 6
        assert_eq!(bucket_index(27), 6); // (27-2)/4 = 6
        // count=28-31
        assert_eq!(bucket_index(28), 6); // (28-2)/4 = 6
        assert_eq!(bucket_index(29), 6); // (29-2)/4 = 6
        assert_eq!(bucket_index(30), 7); // (30-2)/4 = 7
        assert_eq!(bucket_index(100), 7);
    }

    #[test]
    fn test_nn_input_planes_starting_position() {
        // Create a standard starting position with Red to move
        let board = Board::new(RuleSet::Official, 1);

        let (stm, ntm) = NNInputPlanes::from_board(&board);

        // Count non-king pieces: 4 advisors + 4 elephants + 4 horses + 4 chariots + 4 cannons + 10 pawns = 30
        let non_king = count_non_king_pieces(&board);
        assert_eq!(non_king, 30);
        assert_eq!(bucket_index(non_king), 7); // 30 is in bucket 7 (16+ pieces)

        // Verify stm encoding: Red pieces at base 0, Black pieces at base 630
        // Red starts at y=7-9, Black at y=0-2
        // Check a few specific features

        // Red Chariot at (0, 9) - our piece at base 0
        // Piece type Chariot=6, but there's a discrepancy - let me verify piece types
        // Actually from main.rs: King=0, Advisor=1, Elephant=2, Pawn=3, Horse=4, Cannon=5, Chariot=6

        // Verify the stm has our (Red) pieces at base 0
        // Red pieces are at base 0
        let red_chariot_base = 0;
        let red_chariot_pt = PieceType::Chariot as usize;
        let red_chariot_sq = 9 * 9; // y=9, x=0
        let red_chariot_idx = red_chariot_base + red_chariot_pt * 90 + red_chariot_sq;
        assert_eq!(stm.data[red_chariot_idx], 1.0, "Red Chariot at (0,9) should be 1.0 in stm");

        // Black pieces should be at base 630
        let black_chariot_base = 630;
        let black_chariot_pt = PieceType::Chariot as usize;
        let black_chariot_sq = 0; // y=0, x=0 (Black's back rank)
        let black_chariot_idx = black_chariot_base + black_chariot_pt * 90 + black_chariot_sq;
        assert_eq!(stm.data[black_chariot_idx], 1.0, "Black Chariot at (0,0) should be 1.0 in stm");

        // ntm should have opposite perspective
        // For ntm, Black's pieces (their_color = Black when current_side = Red) are at base 0
        // So Black Chariot at (0,0) in ntm (vertically flipped position)
        // After flip: y=0 → y=9
        let ntm_black_chariot_base = 0;
        let ntm_black_chariot_sq = 9 * 9; // y=9, x=0 after flip
        let ntm_black_chariot_idx = ntm_black_chariot_base + black_chariot_pt * 90 + ntm_black_chariot_sq;
        assert_eq!(ntm.data[ntm_black_chariot_idx], 1.0, "Black Chariot flipped to (0,9) should be 1.0 in ntm");

        // For ntm, Red's pieces are at base 630
        let ntm_red_chariot_base = 630;
        let ntm_red_chariot_sq = 0; // y=0, x=0 after flip (Red was at y=9)
        let ntm_red_chariot_idx = ntm_red_chariot_base + red_chariot_pt * 90 + ntm_red_chariot_sq;
        assert_eq!(ntm.data[ntm_red_chariot_idx], 1.0, "Red Chariot flipped to (0,0) should be 1.0 in ntm");

        // Verify total number of active features: 32 pieces total, 2 kings are NOT counted in non_king but still encoded
        // So 32 pieces total = 32 active features in stm
        let stm_active: usize = stm.data.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(stm_active, 32, "Starting position should have 32 pieces encoded");

        // ntm also has 32
        let ntm_active: usize = ntm.data.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(ntm_active, 32, "Starting position should have 32 pieces encoded in ntm");
    }

    #[test]
    fn test_vertical_flip() {
        // Create a board with specific pieces using FEN to test vertical flip.
        // Red Pawn at (4, 6), Black Horse at (1, 3)
        // FEN: rank0/rank1/.../rank9, w = Red to move
        let board = Board::from_fen("9/9/9/1h7/9/9/4P4/9/9/9 w - - 0 1");

        let (stm, ntm) = NNInputPlanes::from_board(&board);

        // Red Pawn at (4, 6) in stm should map to Black's perspective flipped
        // In ntm, our_color = Black (opponent of Red), their_color = Red
        // Red Pawn should be at base 630 in ntm
        let pawn_pt = PieceType::Pawn as usize;

        // stm: Red Pawn at base 0, y=6, x=4 → idx = 0 + 3*90 + 6*9+4 = 270 + 58 = 328
        let stm_pawn_idx = pawn_pt * 90 + 6 * 9 + 4;
        assert_eq!(stm.data[stm_pawn_idx], 1.0, "Red Pawn at (4,6) in stm");

        // ntm: Red Pawn is "their" piece, at base 630, vertically flipped
        // y=6 → y=3 (9-6=3), x=4
        // idx = 630 + 3*90 + 3*9+4 = 630 + 270 + 31 = 931
        let ntm_red_pawn_idx = 630 + pawn_pt * 90 + 3 * 9 + 4;
        assert_eq!(ntm.data[ntm_red_pawn_idx], 1.0, "Red Pawn at (4,6) flipped to (4,3) in ntm at base 630");

        // Black Horse at (1, 3) in stm at base 630
        // idx = 630 + 4*90 + 3*9+1 = 630 + 360 + 28 = 1018
        let horse_pt = PieceType::Horse as usize;
        let stm_horse_idx = 630 + horse_pt * 90 + 3 * 9 + 1;
        assert_eq!(stm.data[stm_horse_idx], 1.0, "Black Horse at (1,3) in stm");

        // ntm: Black Horse is "their" piece (their_color=Black), so at base 0 in ntm
        // Flipped y=3 → y=6, x=1
        // idx = 0 + 4*90 + 6*9+1 = 0 + 360 + 55 = 415
        let ntm_horse_idx = horse_pt * 90 + 6 * 9 + 1;
        assert_eq!(ntm.data[ntm_horse_idx], 1.0, "Black Horse at (1,3) flipped to (1,6) in ntm at base 0");
    }
}
