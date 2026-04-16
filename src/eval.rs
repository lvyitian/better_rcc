// Public items re-exported from eval submodule for use by other modules
// Note: handcrafted_evaluate is imported directly by nn_eval via crate::eval::eval::handcrafted_evaluate

// Re-export types needed by nn_eval (these are pub in main.rs)
#[allow(unused_imports)]
pub use crate::{
    Board, Color, PieceType, Piece, Coord,
    movegen, RuleSet,
};
pub use crate::book::EndgameTablebase;

// Items from crate root - defined at crate level in main.rs (pub const)
use crate::{
    DIRS_4,
    PALACE_DELTAS,
    MATE_SCORE, CHECK_BONUS,
};

pub mod eval_impl {
    use super::*;
    use smallvec::SmallVec;

    // =========================================================================
    // Constants
    // =========================================================================

    // Material values for midgame (MG) and endgame (EG).
    // The engine interpolates between MG and EG based on game phase.
    const MG_VALUE: [i32; 7] = [10000, 135, 105, 80, 350, 500, 650];
    const EG_VALUE: [i32; 7] = [10000, 140, 100, 200, 450, 380, 700];

    // Piece-Square Tables (PST): Position bonuses for each piece type.
    // Each table is a 10×9 array matching board coordinates (y, x).
    // Red uses table directly, Black mirrors y (9-y).
    const MG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 50, 30, 50, 0, 0, 0],
        [0, 0, 0, 30, 20, 30, 0, 0, 0],
        [0, 0, 0, 40, 25, 40, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 50, 30, 50, 0, 0, 0],
        [0, 0, 0, 30, 20, 30, 0, 0, 0],
        [0, 0, 0, 40, 25, 40, 0, 0, 0],
    ];
    const EG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 40, 25, 40, 0, 0, 0],
        [0, 0, 0, 25, 15, 25, 0, 0, 0],
        [0, 0, 0, 30, 20, 30, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 40, 25, 40, 0, 0, 0],
        [0, 0, 0, 25, 15, 25, 0, 0, 0],
        [0, 0, 0, 30, 20, 30, 0, 0, 0],
    ];
    const MG_PST_ADVISOR: [[i32; 9]; 10] = [
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 40, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 40, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
    ];
    const MG_PST_ELEPHANT: [[i32; 9]; 10] = [
        [0, 0, 20, 0, 0, 0, 20, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 30, 0, 20, 0, 30, 0, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 15, 0, 10, 0, 15, 0, 0],
        [0, 0, 15, 0, 10, 0, 15, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 30, 0, 20, 0, 30, 0, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 20, 0, 0, 0, 20, 0, 0],
    ];
    const MG_PST_HORSE: [[i32; 9]; 10] = [
        [5, 10, 20, 20, 20, 20, 20, 10, 5],
        [5, 15, 30, 40, 50, 40, 30, 15, 5],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [20, 40, 60, 80, 90, 80, 60, 40, 20],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [20, 40, 60, 80, 90, 80, 60, 40, 20],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [5, 15, 30, 40, 50, 40, 30, 15, 5],
        [5, 10, 20, 20, 20, 20, 20, 10, 5],
    ];
    const MG_PST_CHARIOT: [[i32; 9]; 10] = [
        [15, 30, 45, 60, 75, 60, 45, 30, 15],
        [30, 45, 60, 75, 90, 75, 60, 45, 30],
        [45, 60, 75, 100, 115, 100, 75, 60, 45],
        [60, 75, 100, 115, 130, 115, 100, 75, 60],
        [45, 60, 75, 100, 115, 100, 75, 60, 45],
        [45, 60, 75, 100, 115, 100, 75, 60, 45],
        [60, 75, 100, 115, 130, 115, 100, 75, 60],
        [45, 60, 75, 100, 115, 100, 75, 60, 45],
        [30, 45, 60, 75, 90, 75, 60, 45, 30],
        [15, 30, 45, 60, 75, 60, 45, 30, 15],
    ];
    const MG_PST_CANNON: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [25, 35, 45, 55, 65, 55, 45, 35, 25],
        [35, 50, 65, 80, 90, 80, 65, 50, 35],
        [45, 60, 75, 90, 100, 90, 75, 60, 45],
        [35, 50, 65, 80, 90, 80, 65, 50, 35],
        [35, 50, 65, 80, 90, 80, 65, 50, 35],
        [45, 60, 75, 90, 100, 90, 75, 60, 45],
        [35, 50, 65, 80, 90, 80, 65, 50, 35],
        [25, 35, 45, 55, 65, 55, 45, 35, 25],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    const MG_PST_PAWN: [[i32; 9]; 10] = [
        [40, 45, 50, 55, 60, 55, 50, 45, 40],
        [35, 40, 45, 50, 55, 50, 45, 40, 35],
        [30, 35, 40, 45, 50, 45, 40, 35, 30],
        [20, 25, 30, 35, 40, 35, 30, 25, 20],
        [15, 18, 20, 25, 30, 25, 20, 18, 15],
        [10, 12, 15, 18, 20, 18, 15, 12, 10],
        [5, 8, 10, 12, 15, 12, 10, 8, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    const EG_PST_PAWN: [[i32; 9]; 10] = [
        [160, 170, 180, 190, 200, 190, 180, 170, 160],
        [150, 160, 170, 180, 190, 180, 170, 160, 150],
        [140, 150, 160, 170, 180, 170, 160, 150, 140],
        [120, 130, 140, 150, 160, 150, 140, 130, 120],
        [100, 110, 120, 130, 140, 130, 120, 110, 100],
        [80, 90, 100, 110, 120, 110, 100, 90, 80],
        [60, 70, 80, 90, 100, 90, 80, 70, 60],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    // Phase weights and total phase
    const PHASE_WEIGHTS: [i32; 7] = [0, 1, 1, 1, 4, 4, 8];
    const TOTAL_PHASE: i32 = 82;

    // =========================================================================
    // Phase and PST
    // =========================================================================

    #[inline(always)]
    pub fn game_phase(board: &Board) -> i32 {
        let mut phase = 0;
        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = board.get(Coord::new(x as i8, y as i8)) {
                    phase += PHASE_WEIGHTS[p.piece_type as usize];
                }
            }
        }
        phase.clamp(0, TOTAL_PHASE)
    }

    #[inline(always)]
    fn pst_val(pt: PieceType, color: Color, x: usize, y: usize, phase: i32) -> i32 {
        let y_mirrored = 9 - y;
        let (y_mg, y_eg) = if color == Color::Red {
            (y, y)
        } else {
            (y_mirrored, y_mirrored)
        };

        let mg = match pt {
            PieceType::King => MG_PST_KING[y_mg][x],
            PieceType::Advisor => MG_PST_ADVISOR[y_mg][x],
            PieceType::Elephant => MG_PST_ELEPHANT[y_mg][x],
            PieceType::Horse => MG_PST_HORSE[y_mg][x],
            PieceType::Chariot => MG_PST_CHARIOT[y_mg][x],
            PieceType::Cannon => MG_PST_CANNON[y_mg][x],
            PieceType::Pawn => MG_PST_PAWN[y_mg][x],
        };

        let eg = match pt {
            PieceType::King => EG_PST_KING[y_eg][x],
            PieceType::Pawn => EG_PST_PAWN[y_eg][x],
            _ => mg,
        };

        // Integer interpolation: (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
        (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    }

    // =========================================================================
    // Helper types and functions
    // =========================================================================

    /// Position + piece type pair for iterating pieces of one color.
    #[derive(Clone, Copy)]
    pub struct PiecePos {
        pub pos: Coord,
        pub pt: PieceType,
    }

    /// Iterate all pieces of one color on the board.
    pub fn pieces_of_color(board: &Board, color: Color) -> SmallVec<[PiecePos; 16]> {
        let mut v = SmallVec::new();
        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(p) = board.get(pos) {
                    if p.color == color {
                        v.push(PiecePos { pos, pt: p.piece_type });
                    }
                }
            }
        }
        v
    }

    /// Get the king's position for a given color.
    pub fn king_pos(board: &Board, color: Color) -> Option<Coord> {
        let (rk, bk) = board.find_kings();
        match color {
            Color::Red => rk,
            Color::Black => bk,
        }
    }

    // =========================================================================
    // STAGE 1: Material + PST
    // =========================================================================

    /// Stage 1: Material value (MG/EG interpolated) + PST bonus for all pieces.
    /// Returns Red-positive score (positive = Red advantage).
    pub fn material_pst(board: &Board) -> i32 {
        let phase = game_phase(board);
        let mut score = 0i32;

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = board.get(pos) {
                    let mg = MG_VALUE[piece.piece_type as usize];
                    let eg = EG_VALUE[piece.piece_type as usize];
                    let val = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
                    score += val * piece.color.sign();

                    let pst = pst_val(piece.piece_type, piece.color, x, y, phase);
                    score += pst;
                }
            }
        }
        score
    }

    // =========================================================================
    // STAGE 2: Mobility (bitboard-accelerated)
    // =========================================================================

    /// Horse mobility: count of horse_attacks() bitboard. Each valid destination = +10.
    /// Returns Red-positive.
    pub fn horse_mobility(board: &Board, pos: Coord, _color: Color) -> i32 {
        let sq = (pos.y * 9 + pos.x) as u8;
        let attacks = board.bitboards.horse_attacks(sq, _color);
        (attacks.count_ones() as i32) * 10
    }

    /// Chariot mobility: empty squares in 4 cardinal directions. Each = +5.
    /// Returns Red-positive.
    pub fn chariot_mobility(board: &Board, pos: Coord, _color: Color) -> i32 {
        let sq = (pos.y * 9 + pos.x) as u8;
        let attacks = board.bitboards.chariot_attacks(sq, _color);
        let empty = attacks & !board.bitboards.occupied_all();
        (empty.count_ones() as i32) * 5
    }

    /// Elephant mobility: number of valid elephant endpoints. Each = +5.
    /// Returns Red-positive.
    pub fn elephant_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
        let moves = movegen::generate_elephant_moves(board, pos, color);
        (moves.len() as i32) * 5
    }

    /// Cannon activity: count platform pieces (advisor/elephant/pawn) between cannon
    /// and first piece in each direction. Each platform = +10.
    /// Returns Red-positive.
    pub fn cannon_activity(board: &Board, pos: Coord, color: Color) -> i32 {
        let mut score = 0i32;
        let occ_all = board.bitboards.occupied_all();

        for &(dx, dy) in DIRS_4.iter() {
            let mut cx = pos.x + dx;
            let mut cy = pos.y + dy;
            while (0..9).contains(&cx) && (0..10).contains(&cy) {
                let c = Coord::new(cx, cy);
                let c_sq = (cy as i8 * 9 + cx as i8) as u8;
                if occ_all & (1_u128 << c_sq) != 0 {
                    if let Some(p) = board.get(c) {
                        if p.color == color {
                            if p.piece_type == PieceType::Advisor
                                || p.piece_type == PieceType::Elephant
                                || p.piece_type == PieceType::Pawn
                            {
                                score += 10;
                            }
                        }
                    }
                    break;
                }
                cx += dx;
                cy += dy;
            }
        }
        score
    }

    // =========================================================================
    // STAGE 2: Pawn Structure
    // =========================================================================

    /// Pawn structure evaluation. Returns Red-positive score.
    pub fn pawn_structure(board: &Board, color: Color, _phase: i32) -> i32 {
        let mut score = 0i32;
        let mut pawns_per_file = [0i32; 9];

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(p) = board.get(pos)
                    && p.color == color && p.piece_type == PieceType::Pawn
                {
                    pawns_per_file[x] += 1;

                    // Doubled penalty: -30 for each extra pawn beyond the first
                    if pawns_per_file[x] >= 2 {
                        score -= 30;
                    }

                    // Back-rank penalty
                    let on_starting_rank = match color {
                        Color::Red => y == 6,
                        Color::Black => y == 3,
                    };
                    if on_starting_rank {
                        score -= 20;
                    }

                    // Horizontal link bonus
                    let left = Coord::new(pos.x - 1, pos.y);
                    let right = Coord::new(pos.x + 1, pos.y);
                    if let Some(lp) = board.get(left)
                        && lp.color == color && lp.piece_type == PieceType::Pawn
                    {
                        score += 15;
                    }
                    if let Some(rp) = board.get(right)
                        && rp.color == color && rp.piece_type == PieceType::Pawn
                    {
                        score += 15;
                    }

                    // Advancement bonus (crossed river)
                    let crossed = pos.crosses_river(color);
                    if crossed {
                        let advancement = match color {
                            Color::Red => 6 - y,
                            Color::Black => y - 3,
                        };
                        if advancement > 0 {
                            score += advancement * 5;
                        }
                        let on_enemy_back = match color {
                            Color::Red => y == 0,
                            Color::Black => y == 9,
                        };
                        if on_enemy_back {
                            score += 80;
                        }
                    }

                    // Central file bonus (files 4-5)
                    if pos.x == 4 || pos.x == 5 {
                        let ranks_advanced = match color {
                            Color::Red => 6 - y,
                            Color::Black => y - 3,
                        };
                        if ranks_advanced > 0 {
                            score += ranks_advanced * 10;
                        }
                    }
                }
            }
        }
        score * color.sign()
    }

    // =========================================================================
    // STAGE 2: Elephant Structure
    // =========================================================================

    /// Elephant structure evaluation. Returns Red-positive score.
    pub fn elephant_structure(board: &Board, color: Color, phase: i32) -> i32 {
        let count = board.bitboards.piece_bitboard(PieceType::Elephant, color).count_ones() as i32;
        let mg_factor = phase;
        let eg_factor = TOTAL_PHASE - phase;

        let penalty = match count {
            0 => 80 * mg_factor + 200 * eg_factor,
            1 => 40 * mg_factor + 100 * eg_factor,
            _ => 0,
        };
        // Correct formula: -(penalty / TOTAL_PHASE) * (-color.sign())
        // Red (sign=+1): -(penalty/TOTAL_PHASE) * (-1) = -(penalty/TOTAL_PHASE) [subtracts from Red score]
        // Black (sign=-1): -(penalty/TOTAL_PHASE) * (+1) = +(penalty/TOTAL_PHASE) [adds to Red score]
        -(penalty / TOTAL_PHASE) * (-color.sign())
    }

    // =========================================================================
    // STAGE 3: King Safety
    // =========================================================================

    /// King safety for one side. Returns Red-positive score.
    pub fn king_safety(board: &Board, king_color: Color, phase: i32) -> Option<i32> {
        let king = king_pos(board, king_color)?;
        let opponent = king_color.opponent();
        let mut score = 0i32;

        // Palace defenders
        for delta in PALACE_DELTAS {
            let pos = Coord::new(king.x + delta.0, king.y + delta.1);
            if pos.is_valid() && pos.in_palace(king_color) {
                if let Some(p) = board.get(pos) && p.color == king_color {
                    score += match p.piece_type {
                        PieceType::Advisor => 15,
                        PieceType::Elephant => 10,
                        _ => 0,
                    };
                }
            }
        }

        // Enemy piece pressure (distance-weighted)
        let mg_factor = phase;
        let occ = board.bitboards.occupied_all();
        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                let sq = (y * 9 + x) as u8;
                if occ & (1_u128 << sq) == 0 {
                    continue;
                }
                if let Some(p) = board.get(pos) && p.color == opponent {
                    let dist = (pos.x - king.x).abs() + (pos.y - king.y).abs();
                    let threat = match p.piece_type {
                        PieceType::Chariot => ((14 - dist).max(0) as i32) * 8,
                        PieceType::Cannon => ((12 - dist).max(0) as i32) * 5,
                        PieceType::Horse => ((10 - dist).max(0) as i32) * 6,
                        PieceType::Pawn if dist <= 4 => ((5 - dist) as i32) * 6,
                        _ => 0,
                    };
                    score -= threat * mg_factor / TOTAL_PHASE;
                }
            }
        }

        // Attackers on enemy king (uses attackers() bitboard)
        if let Some(ek) = king_pos(board, opponent) {
            let ek_sq = (ek.y * 9 + ek.x) as u8;
            let attackers_bb = board.bitboards.attackers(ek_sq, king_color);
            let attack_count = attackers_bb.count_ones() as i32;
            score += attack_count * 15;
        }

        Some(score * king_color.sign())
    }

    // =========================================================================
    // STAGE 3: Hanging Pieces
    // =========================================================================

    /// True if attacker at `from` can attack `to` given attacker piece type and color.
    /// All pieces use Manhattan (taxi) distance: |dx| + |dy|
    /// - Chariot: same row/col, clear path
    /// - Cannon: same row/col, exactly 1 screen
    /// - Horse: L-move, Manhattan = 3
    /// - Pawn: ORTHOGONAL forward (both colors), or forward+sideways after crossing river
    /// - Advisor: diagonal ±1, in palace
    /// - Elephant: diagonal ±2, eye empty, doesn't cross river
    /// - King: orthogonal ±1, in palace
    fn can_attack(board: &Board, from: Coord, to: Coord, pt: PieceType, color: Color) -> bool {
        let dx = (to.x - from.x).abs() as i32;
        let dy = (to.y - from.y).abs() as i32;
        let dist = dx + dy;
        match pt {
            PieceType::Chariot => {
                if from.y == to.y {
                    let step = if from.x < to.x { 1 } else { -1 };
                    let mut x = from.x + step;
                    while x != to.x {
                        if board.get(Coord::new(x, from.y)).is_some() { return false; }
                        x += step;
                    }
                    true
                } else if from.x == to.x {
                    let step = if from.y < to.y { 1 } else { -1 };
                    let mut y = from.y + step;
                    while y != to.y {
                        if board.get(Coord::new(from.x, y)).is_some() { return false; }
                        y += step;
                    }
                    true
                } else {
                    false
                }
            }
            PieceType::Cannon => {
                if from.y == to.y {
                    let step = if from.x < to.x { 1 } else { -1 };
                    let mut screens = 0;
                    let mut x = from.x + step;
                    while x != to.x {
                        if board.get(Coord::new(x, from.y)).is_some() { screens += 1; }
                        x += step;
                    }
                    screens == 1
                } else if from.x == to.x {
                    let step = if from.y < to.y { 1 } else { -1 };
                    let mut screens = 0;
                    let mut y = from.y + step;
                    while y != to.y {
                        if board.get(Coord::new(from.x, y)).is_some() { screens += 1; }
                        y += step;
                    }
                    screens == 1
                } else {
                    false
                }
            }
            PieceType::Horse => dist == 3, // L-move: (2,1) or (1,2) → taxi = 3
            PieceType::Pawn => {
                // Chinese chess pawns capture ORTHOGONALLY (not diagonally)
                let forward_dir = if color == Color::Red { -1 } else { 1 };
                let raw_dx = to.x - from.x;
                let raw_dy = to.y - from.y;
                let is_forward = raw_dy == forward_dir;
                let is_sideways = raw_dx.abs() == 1 && raw_dy == 0;
                let crossed = from.crosses_river(color);
                (is_forward || (crossed && is_sideways)) && dist == 1
            }
            PieceType::Advisor => {
                // Advisor: diagonal ±1 step, must be in palace
                dist == 2 && to.in_palace(color)
            }
            PieceType::Elephant => {
                // Elephant: diagonal ±2 steps, eye empty, doesn't cross river
                if dist != 4 || to.crosses_river(color) { return false; }
                let eye_x = from.x + (to.x - from.x) / 2;
                let eye_y = from.y + (to.y - from.y) / 2;
                board.get(Coord::new(eye_x, eye_y)).is_none()
            }
            PieceType::King => {
                // King: orthogonal ±1, must be in palace
                dist == 1 && to.in_palace(color)
            }
        }
    }

    /// True if defender at `from` can defend `to` (same geometry as can_attack).
    fn can_defend(board: &Board, from: Coord, to: Coord, pt: PieceType, color: Color) -> bool {
        let dx = (to.x - from.x).abs() as i32;
        let dy = (to.y - from.y).abs() as i32;
        let dist = dx + dy;
        match pt {
            PieceType::Chariot | PieceType::Cannon => from.y == to.y || from.x == to.x,
            PieceType::Horse => dist == 3,
            PieceType::Pawn => {
                let forward_dir = if color == Color::Red { -1 } else { 1 };
                let raw_dx = to.x - from.x;
                let raw_dy = to.y - from.y;
                let is_forward = raw_dy == forward_dir;
                let is_sideways = raw_dx.abs() == 1 && raw_dy == 0;
                let crossed = from.crosses_river(color);
                (is_forward || (crossed && is_sideways)) && dist == 1
            }
            PieceType::Advisor => dist == 2 && to.in_palace(color),
            PieceType::Elephant => {
                if dist != 4 || to.crosses_river(color) { return false; }
                let eye_x = from.x + (to.x - from.x) / 2;
                let eye_y = from.y + (to.y - from.y) / 2;
                board.get(Coord::new(eye_x, eye_y)).is_none()
            }
            PieceType::King => dist == 1 && to.in_palace(color),
        }
    }

    /// Detect hanging and overloaded enemy pieces. Returns Red-positive score.
    pub fn hanging_pieces(board: &Board, color: Color, phase: i32) -> i32 {
        let mut score = 0i32;
        let mg_factor = phase;

        let our = pieces_of_color(board, color);
        let enemy = pieces_of_color(board, color.opponent());

        for enemy_piece in &enemy {
            let mut attackers = 0;
            let mut defenders = 0;

            for our_piece in &our {
                if can_attack(board, our_piece.pos, enemy_piece.pos, our_piece.pt, color) {
                    attackers += 1;
                }
            }
            for def_piece in enemy.iter() {
                if def_piece.pos == enemy_piece.pos { continue; }
                if can_defend(board, def_piece.pos, enemy_piece.pos, def_piece.pt, color.opponent()) {
                    defenders += 1;
                }
            }

            let val = MG_VALUE[enemy_piece.pt as usize];
            if attackers > 0 && defenders == 0 {
                score += (val * 30 / 100) * mg_factor / TOTAL_PHASE;
            } else if attackers > 1 && defenders == 1 {
                score += (val * 15 / 100) * mg_factor / TOTAL_PHASE;
            }
        }

        score * color.sign()
    }

    // =========================================================================
    // MAIN ENTRY POINT
    // =========================================================================

    pub fn handcrafted_evaluate(board: &Board, side: Color, _initiative: bool) -> i32 {
        // Checkmate detection
        let (rk, bk) = board.find_kings();
        if rk.is_none() {
            let fallback = if side == Color::Red { -MATE_SCORE } else { MATE_SCORE };
            if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
                if conf < 0.2 { return tb_score; }
                let weight = conf.max(0.3);
                return (tb_score as f32 * weight + fallback as f32 * (1.0 - weight)) as i32;
            }
            return fallback;
        }
        if bk.is_none() {
            let fallback = if side == Color::Red { MATE_SCORE } else { -MATE_SCORE };
            if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
                if conf < 0.2 { return tb_score; }
                let weight = conf.max(0.3);
                return (tb_score as f32 * weight + fallback as f32 * (1.0 - weight)) as i32;
            }
            return fallback;
        }

        let phase = game_phase(board);
        let mut raw_score = 0i32;

        // Stage 1: Material + PST
        raw_score += material_pst(board);

        // Stage 2: Mobility per piece
        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = board.get(pos) {
                    let sign = piece.color.sign();
                    match piece.piece_type {
                        PieceType::Horse => {
                            raw_score += horse_mobility(board, pos, piece.color) * sign;
                        }
                        PieceType::Chariot => {
                            raw_score += chariot_mobility(board, pos, piece.color) * sign;
                        }
                        PieceType::Cannon => {
                            raw_score += cannon_activity(board, pos, piece.color) * sign;
                        }
                        PieceType::Elephant => {
                            raw_score += elephant_mobility(board, pos, piece.color) * sign;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Pawn structure
        raw_score += pawn_structure(board, Color::Red, phase);
        raw_score += pawn_structure(board, Color::Black, phase);

        // Elephant structure
        raw_score += elephant_structure(board, Color::Red, phase);
        raw_score += elephant_structure(board, Color::Black, phase);

        // Stage 3: King safety
        if let Some(ks) = king_safety(board, Color::Red, phase) {
            raw_score += ks;
        }
        if let Some(ks) = king_safety(board, Color::Black, phase) {
            raw_score += ks;
        }

        // Hanging pieces
        raw_score += hanging_pieces(board, Color::Red, phase);
        raw_score += hanging_pieces(board, Color::Black, phase);

        // Stage 4: Check bonus
        if board.is_check(Color::Black) {
            raw_score += CHECK_BONUS;
        }
        if board.is_check(Color::Red) {
            raw_score -= CHECK_BONUS;
        }

        let final_score = raw_score * side.sign();

        // Blend with tablebase if available
        if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
            if conf < 0.2 { return tb_score; }
            let weight = conf.max(0.3);
            return (tb_score as f32 * weight + final_score as f32 * (1.0 - weight)) as i32;
        }

        final_score
    }
}
