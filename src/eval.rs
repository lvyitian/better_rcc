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
    BOARD_WIDTH, BOARD_HEIGHT,
    CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX,
    HORSE_DELTAS, HORSE_BLOCKS, DIRS_4,
    PALACE_DELTAS,
    MATE_SCORE, CHECK_BONUS,
};

pub mod eval_impl {
    use super::*;
    use smallvec::SmallVec;

    // Material values for midgame (opening/middlegame) and endgame (simplified positions).
    // The engine interpolates between MG and EG based on game phase.
    //
    // Piece values (MG/EG):
    // - King (10000/10000) - invaluable, must protect
    // - Advisor (135/140) - protects king directly, palace-bound but critical
    // - Elephant (105/100) - river-restricted, less flexible than Advisor
    // - Pawn (80/200) - weak but advances; strong in EG when passed
    // - Horse (350/450) - mobility, requires coordination; EG surge per Fairy-Stockfish
    // - Cannon (500/380) - screen attack, strong throughout game; MG > EG per Fairy-Stockfish
    // - Chariot (650/700) - strongest piece, rook-like
    const MG_VALUE: [i32; 7] = [10000, 135, 105, 80, 350, 500, 650];
    const EG_VALUE: [i32; 7] = [10000, 140, 100, 200, 450, 380, 700];

    // Piece-Square Tables (PST): Position bonuses for each piece type.
    // Each table is a 10×9 array matching board coordinates (y, x).
    // Higher = better square. Red uses table directly, Black mirrors y (9-y).
    //
    // King PST: Palace corners (x=3,5; y=0,2) are safest - only 3 approach squares within palace.
    // Palace center (x=4, y=1) is more exposed - faces threats from multiple directions.
    // MG: King stays in palace corners. EG: King still confined but slightly more central.
    //
    // Palace approach vectors analysis:
    // - (3,0)/(5,0): back corners - only 2 approach squares within palace (board edge protects)
    // - (4,1): center - 4 approach squares (most exposed)
    // - (3,1)/(5,1): side centers - 3 approach squares each
    // - (4,0)/(4,2): back/front centers - 3 approach squares
    //
    // Value ranking: corners > side centers > back/front centers > center
    // y=0 (back edge) > y=2 (front) > y=1 (center row) for same x position
    const MG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 50, 30, 50, 0, 0, 0],  // y=0: back corners (50) > back center (30)
        [0, 0, 0, 30, 20, 30, 0, 0, 0],  // y=1: side centers (30) > center (20)
        [0, 0, 0, 40, 25, 40, 0, 0, 0],  // y=2: front corners (40) > front center (25)
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 50, 30, 50, 0, 0, 0],  // Red palace (mirrored)
        [0, 0, 0, 30, 20, 30, 0, 0, 0],
        [0, 0, 0, 40, 25, 40, 0, 0, 0],
    ];

    // EG King: Lower values overall, same spatial pattern as MG
    const EG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 40, 25, 40, 0, 0, 0],  // y=0: back corners (40) > back center (25)
        [0, 0, 0, 25, 15, 25, 0, 0, 0],  // y=1: side centers (25) > center (15)
        [0, 0, 0, 30, 20, 30, 0, 0, 0],  // y=2: front corners (30) > front center (20)
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 40, 25, 40, 0, 0, 0],  // Red palace (mirrored)
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

    // Elephant PST: Prefers back-rank diagonal positions, not river.
    // River positions (y=4,5) are EXPOSED (high center, can't cross) - lower is better.
    // Back rank (y=0,2,7,9) and near-river (y=3) positions provide good defense.
    const MG_PST_ELEPHANT: [[i32; 9]; 10] = [
        [0, 0, 20, 0, 0, 0, 20, 0, 0],  // Back rank corners - best defensive spots
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 30, 0, 20, 0, 30, 0, 10],  // Near-back-rank diagonals
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 15, 0, 10, 0, 15, 0, 0],  // River - EXPOSED, lower value
        [0, 0, 15, 0, 10, 0, 15, 0, 0],  // River - EXPOSED, lower value
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 30, 0, 20, 0, 30, 0, 10],  // Mirror of near-back-rank
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 20, 0, 0, 0, 20, 0, 0],  // Mirror of back rank corners
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

    // Chariot PST: Scaled up to ~20% of piece value (was ~14%).
    // Center (x=4,y=3,6) max 130 ≈ 20% of MG value (650).
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

    // Cannon PST: Scaled to ~20% of MG value (500). Max center = 100.
    // y=1,8 (palace-entrance) elevated to match new scale.
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

    // Pawn PST: Lower values to avoid irrational pawn fixation.
    // Max MG (~60) ≈ 17% of Horse value (350), Max EG (~120) ≈ 43% of Horse EG (280).
    // Maintains relative ranking: forward squares > backward squares.
    const MG_PST_PAWN: [[i32; 9]; 10] = [
        [40, 45, 50, 55, 60, 55, 50, 45, 40],   // y=0: enemy back rank - max 60
        [35, 40, 45, 50, 55, 50, 45, 40, 35],   // y=1
        [30, 35, 40, 45, 50, 45, 40, 35, 30],   // y=2: advanced
        [20, 25, 30, 35, 40, 35, 30, 25, 20],   // y=3: crossed river
        [15, 18, 20, 25, 30, 25, 20, 18, 15],   // y=4: near river
        [10, 12, 15, 18, 20, 18, 15, 12, 10],   // y=5: at river
        [5, 8, 10, 12, 15, 12, 10, 8, 5],       // y=6: before river
        [0, 0, 0, 0, 0, 0, 0, 0, 0],            // y=7: starting
        [0, 0, 0, 0, 0, 0, 0, 0, 0],            // y=8
        [0, 0, 0, 0, 0, 0, 0, 0, 0],            // y=9: back rank
    ];

    const EG_PST_PAWN: [[i32; 9]; 10] = [
        // y=0: max 200 — scales with pawn's EG value surge (80→200 = 2.5×)
        [160, 170, 180, 190, 200, 190, 180, 170, 160],
        [150, 160, 170, 180, 190, 180, 170, 160, 150],
        [140, 150, 160, 170, 180, 170, 160, 150, 140],
        [120, 130, 140, 150, 160, 150, 140, 130, 120],  // y=3: crossed river
        [100, 110, 120, 130, 140, 130, 120, 110, 100],  // y=4: near river
        [80, 90, 100, 110, 120, 110, 100, 90, 80],
        [60, 70, 80, 90, 100, 90, 80, 70, 60],          // y=6: before river
        [0, 0, 0, 0, 0, 0, 0, 0, 0],                    // y=7: starting
        [0, 0, 0, 0, 0, 0, 0, 0, 0],                    // y=8
        [0, 0, 0, 0, 0, 0, 0, 0, 0],                    // y=9: back rank
    ];

    // Phase weights: King=0, Advisor=1, Elephant=1, Pawn=1, Horse=4, Cannon=4, Chariot=8
    // Per side max: 2*1 + 2*1 + 5*1 + 2*4 + 2*4 + 2*8 = 41 (King excluded)
    const PHASE_WEIGHTS: [i32; 7] = [0, 1, 1, 1, 4, 4, 8];
    const TOTAL_PHASE: i32 = 82; // 41 per side × 2 sides

    #[inline(always)]
    pub fn game_phase(board: &Board) -> i32 {
        use crate::Bitboards;
        let mut phase = 0;
        let mut bb = board.bitboards.occupied_all();
        while bb != 0 {
            let sq = Bitboards::lsb_index(bb);
            if let Some(p) = board.bitboards.piece_at(sq) {
                phase += PHASE_WEIGHTS[p.piece_type as usize];
            }
            bb &= bb - 1;
        }
        phase.clamp(0, TOTAL_PHASE)
    }

    #[inline(always)]
    fn pst_val(
        pt: PieceType,
        color: Color,
        x: usize,
        y: usize,
        phase: i32,
    ) -> i32 {
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

    /// Simplified O(1) center control evaluation.
    ///
    /// Counts weighted attacks on the core area (x=3-5, y=3-6) using bitboard
    /// attack methods and CORE_AREA_MASK.
    ///
    /// Weights: chariot/cannon=2, horse/pawn=1
    /// Net attacks are scaled by 5 and phase factor.
    fn center_control(board: &Board, color: Color, phase: i32) -> i32 {
        use crate::Bitboards;
        use crate::bitboards::CORE_AREA_MASK;

        let bitboards = &board.bitboards;
        let mut our_attacks = 0;
        let mut their_attacks = 0;

        // Iterate over all pieces via bitboards
        let mut occ = bitboards.occupied_all();
        while occ != 0 {
            let sq = Bitboards::lsb_index(occ);
            if let Some(piece) = bitboards.piece_at(sq) {
                let weight = match piece.piece_type {
                    PieceType::Chariot | PieceType::Cannon => 2,
                    PieceType::Horse | PieceType::Pawn => 1,
                    _ => 0,
                };

                if weight > 0 {
                    let attacks = match piece.piece_type {
                        PieceType::Chariot => bitboards.chariot_attacks(sq, piece.color),
                        PieceType::Cannon => bitboards.cannon_attacks(sq, piece.color),
                        PieceType::Horse => bitboards.horse_attacks(sq, piece.color),
                        PieceType::Pawn => bitboards.pawn_attacks(sq, piece.color),
                        _ => 0,
                    };
                    let core_attacks = (attacks & CORE_AREA_MASK).count_ones() as i32;
                    if core_attacks > 0 {
                        if piece.color == color {
                            our_attacks += core_attacks * weight;
                        } else {
                            their_attacks += core_attacks * weight;
                        }
                    }
                }
            }
            occ &= occ - 1;
        }

        let net_attacks = (our_attacks - their_attacks) * color.sign();
        net_attacks * 5 * (30 + 70 * phase) / 100
    }

/// Evaluate attack rewards: hanging pieces, overload, central attacks, king attacks
/// Motivates aggressive play by rewarding actual threats
fn attack_rewards(board: &Board, color: Color, phase: i32) -> i32 {
    let mut score = 0;
    let mg_factor = phase;
    let eg_factor = TOTAL_PHASE - phase;

    // Collect our piece positions and their attack targets
    #[derive(Clone)]
    struct PieceInfo {
        pos: Coord,
        pt: PieceType,
    }
    let mut our_pieces: SmallVec::<[PieceInfo; 16]> = SmallVec::new();
    let mut enemy_pieces: SmallVec::<[PieceInfo; 16]> = SmallVec::new();
    use crate::Bitboards;
    let bitboards = &board.bitboards;

    let mut occ = bitboards.occupied(color);
    while occ != 0 {
        let sq = Bitboards::lsb_index(occ);
        if let Some(p) = bitboards.piece_at(sq) {
            let x = sq % 9;
            let y = sq / 9;
            our_pieces.push(PieceInfo { pos: Coord::new(x as i8, y as i8), pt: p.piece_type });
        }
        occ &= occ - 1;
    }

    let opp = color.opponent();
    let mut occ_enemy = bitboards.occupied(opp);
    while occ_enemy != 0 {
        let sq = Bitboards::lsb_index(occ_enemy);
        if let Some(p) = bitboards.piece_at(sq) {
            let x = sq % 9;
            let y = sq / 9;
            enemy_pieces.push(PieceInfo { pos: Coord::new(x as i8, y as i8), pt: p.piece_type });
        }
        occ_enemy &= occ_enemy - 1;
    }

    // For each enemy piece, count how many attacks and defenses it has
    for (idx, enemy_piece) in enemy_pieces.iter().enumerate() {
        let mut attacked_by_us = 0;
        let mut defended_by_them = 0;

        for our_piece in &our_pieces {
            // Check if we can attack enemy_piece's position
            // Simple check: can any of our pieces "see" this square?
            // For sliding pieces (chariot/cannon), need clear line of sight
            // For non-sliding, just check distance
            let dist = our_piece.pos.distance_to(enemy_piece.pos);

            let can_attack = match our_piece.pt {
                PieceType::Chariot => {
                    // Chariot can attack if on same row/col with clear path
                    let same_row = our_piece.pos.y == enemy_piece.pos.y;
                    let same_col = our_piece.pos.x == enemy_piece.pos.x;
                    if same_row || same_col {
                        // Check path is clear
                        let (start, end) = if same_row {
                            let x_min = our_piece.pos.x.min(enemy_piece.pos.x);
                            let x_max = our_piece.pos.x.max(enemy_piece.pos.x);
                            (x_min + 1, x_max - 1)
                        } else {
                            let y_min = our_piece.pos.y.min(enemy_piece.pos.y);
                            let y_max = our_piece.pos.y.max(enemy_piece.pos.y);
                            (y_min + 1, y_max - 1)
                        };
                        let mut clear = true;
                        // Simple path check - just ensure empty between
                        // For same row: check x between
                        // For same col: check y between
                        if same_row {
                            for check_x in start..=end {
                                let c = Coord::new(check_x, our_piece.pos.y);
                                if board.get(c).is_some() && c != enemy_piece.pos {
                                    clear = false;
                                    break;
                                }
                            }
                        } else {
                            for check_y in start..=end {
                                let c = Coord::new(our_piece.pos.x, check_y);
                                if board.get(c).is_some() && c != enemy_piece.pos {
                                    clear = false;
                                    break;
                                }
                            }
                        }
                        clear
                    } else {
                        false
                    }
                },
                PieceType::Cannon => {
                    // Cannon needs exactly 1 screen between
                    let same_row = our_piece.pos.y == enemy_piece.pos.y;
                    let same_col = our_piece.pos.x == enemy_piece.pos.x;
                    if same_row || same_col {
                        let mut screen_count = 0;
                        if same_row {
                            let x_min = our_piece.pos.x.min(enemy_piece.pos.x);
                            let x_max = our_piece.pos.x.max(enemy_piece.pos.x);
                            for check_x in (x_min+1)..(x_max) {
                                let c = Coord::new(check_x, our_piece.pos.y);
                                if board.get(c).is_some() && c != enemy_piece.pos {
                                    screen_count += 1;
                                }
                            }
                        } else {
                            let y_min = our_piece.pos.y.min(enemy_piece.pos.y);
                            let y_max = our_piece.pos.y.max(enemy_piece.pos.y);
                            for check_y in (y_min+1)..(y_max) {
                                let c = Coord::new(our_piece.pos.x, check_y);
                                if board.get(c).is_some() && c != enemy_piece.pos {
                                    screen_count += 1;
                                }
                            }
                        }
                        screen_count == 1
                    } else {
                        false
                    }
                },
                PieceType::Horse => dist <= 2,  // Horse can be within 2
                PieceType::Pawn => {
                    // Pawn attacks adjacent squares only (forward and side diagonals)
                    // Only count dist==1: pawn can only reach adjacent diagonal squares
                    dist == 1
                },
                _ => dist == 1,  // Advisors, elephants, king: adjacent only
            };

            if can_attack {
                attacked_by_us += 1;
            }
        }

        // Count their defenders (excluding the piece itself)
        for (def_idx, their_piece) in enemy_pieces.iter().enumerate() {
            if def_idx == idx { continue; }  // Don't count self-defense
            let dist = their_piece.pos.distance_to(enemy_piece.pos);
            let can_defend = match their_piece.pt {
                PieceType::Chariot | PieceType::Cannon => {
                    // Same line check as attack
                    let same_row = their_piece.pos.y == enemy_piece.pos.y;
                    let same_col = their_piece.pos.x == enemy_piece.pos.x;
                    same_row || same_col
                },
                PieceType::Horse => dist <= 2,
                PieceType::Pawn => dist == 1,
                _ => dist == 1,
            };
            if can_defend {
                defended_by_them += 1;
            }
        }

        // Hanging piece: attacked with no defense, or attacked with 1 defense but we're attacking with multiple
        if attacked_by_us > 0 && defended_by_them == 0 {
            let val = (30 * mg_factor + 20 * eg_factor) / TOTAL_PHASE;
            score += val;
        } else if attacked_by_us > 1 && defended_by_them == 1 {
            let val = (30 * mg_factor + 20 * eg_factor) / TOTAL_PHASE;
            score += val / 2;  // Partial bonus for overloaded defender
        }
    }

    // Central attack bonus: pieces attacking core area (x ∈ [3,5], y ∈ [3,6])
    let mut central_attacks = 0;
    for our_piece in &our_pieces {
        match our_piece.pt {
            PieceType::Chariot | PieceType::Cannon => {
                // Count attacks on core: piece shares row or column with core area
                let same_row = our_piece.pos.y >= CORE_Y_MIN && our_piece.pos.y <= CORE_Y_MAX;
                let same_col = our_piece.pos.x >= CORE_X_MIN && our_piece.pos.x <= CORE_X_MAX;
                if same_row || same_col {
                    central_attacks += 1;
                }
            },
            PieceType::Horse | PieceType::Pawn => {
                // Check if piece position is in core
                let in_core = our_piece.pos.x >= CORE_X_MIN && our_piece.pos.x <= CORE_X_MAX
                    && our_piece.pos.y >= CORE_Y_MIN && our_piece.pos.y <= CORE_Y_MAX;
                if in_core {
                    central_attacks += 1;
                }
            },
            _ => {}
        }
    }
    let central_bonus = central_attacks * (10 * mg_factor + 5 * eg_factor) / TOTAL_PHASE;
    score += central_bonus;

    score * color.sign()
}

    fn piece_coordination(board: &Board, color: Color, phase: i32) -> i32 {
        use crate::Bitboards;
        let mut coordination = 0;
        let (_rk, _bk) = board.find_kings();
        let enemy_king = match color {
            Color::Red => _bk,
            Color::Black => _rk,
        };
        let bitboards = &board.bitboards;

        // Collect our pieces via bitboards
        let mut our_pieces = SmallVec::<[(Piece, Coord); 16]>::new();
        let mut occ = bitboards.occupied(color);
        while occ != 0 {
            let sq = Bitboards::lsb_index(occ);
            if let Some(piece) = bitboards.piece_at(sq) {
                let x = sq % 9;
                let y = sq / 9;
                our_pieces.push((piece, Coord::new(x as i8, y as i8)));
            }
            occ &= occ - 1;
        }

        let mut has_horse = false;
        let mut has_chariot = false;
        let mut horse_in_attack = false;
        let mut chariot_in_support = false;

        for (piece, pos) in &our_pieces {
            match piece.piece_type {
                PieceType::Horse => {
                    has_horse = true;
                    if let Some(ek) = enemy_king {
                        let dist = pos.distance_to(ek);
                        if dist <= 3 && pos.crosses_river(color) {
                            horse_in_attack = true;
                        }
                    }
                }
                PieceType::Chariot => {
                    has_chariot = true;
                    if pos.in_core_area() {
                        chariot_in_support = true;
                    }
                }
                _ => {}
            }
        }

        if has_horse && has_chariot && horse_in_attack && chariot_in_support {
            coordination += 50 * color.sign();
        }

        // Cannon platform detection: simplified direct position checks
        // A cannon gains value when it has advisor, elephant, or pawn pieces nearby as screens
        for (piece, pos) in &our_pieces {
            if piece.piece_type == PieceType::Cannon {
                let mut platform_count = 0;
                for (dx, dy) in DIRS_4 {
                    let tar = Coord::new(pos.x + dx, pos.y + dy);
                    if tar.is_valid()
                        && let Some(p) = board.get(tar)
                        && (p.piece_type == PieceType::Advisor
                            || p.piece_type == PieceType::Elephant
                            || p.piece_type == PieceType::Pawn)
                    {
                        platform_count += 1;
                    }
                }
                // Bonus for each platform piece found (cap at 2 per direction)
                if platform_count > 0 {
                    coordination += platform_count * 10 * color.sign();
                }
            }
        }

        coordination * (100 + 30 * (TOTAL_PHASE - phase)) / 100
    }

    fn horse_mobility(board: &Board, pos: Coord, _color: Color) -> i32 {
        // Horse mobility: number of valid jumps × 10
        // A valid jump has empty horse-head position and can land anywhere (enemy or empty)
        let mut mobility = 0;

        for i in 0..8 {
            let (dx, dy) = HORSE_DELTAS[i];
            let (bx, by) = HORSE_BLOCKS[i];
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            let block = Coord::new(pos.x + bx, pos.y + by);

            // Valid jump: target is on board AND horse head is empty
            if tar.is_valid() && board.get(block).is_none() {
                mobility += 1;
            }
        }

        mobility * 10
    }

    fn cannon_support(board: &Board, pos: Coord, color: Color) -> i32 {
        let mut support = 0;

        for (dx, dy) in DIRS_4 {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;
            let mut platform_found = false;

            while (0..BOARD_WIDTH).contains(&x) && (0..BOARD_HEIGHT).contains(&y) {
                let tar = Coord::new(x, y);
                if board.get(tar).is_some() {
                    if !platform_found {
                        platform_found = true;
                    } else {
                        if board.get(tar).unwrap().color != color {
                            support += 10;
                        }
                        break;
                    }
                }
                x += dx;
                y += dy;
            }
        }
        support
    }

    fn king_safety(board: &Board, color: Color, phase: i32) -> Option<i32> {
        use crate::Bitboards;
        let (rk, bk) = board.find_kings();
        let king_pos = match color {
            Color::Red => rk?,
            Color::Black => bk?,
        };
        let opponent = color.opponent();
        let mut safety = 0;
        let bitboards = &board.bitboards;

        for (dx, dy) in PALACE_DELTAS {
            let pos = Coord::new(king_pos.x + dx, king_pos.y + dy);
            if pos.is_valid() && pos.in_palace(color)
                && let Some(p) = board.get(pos)
                    && p.color == color {
                        safety += match p.piece_type {
                            PieceType::Advisor => 25,
                            PieceType::Elephant => 15,
                            _ => 5,
                        };
                    }
        }

        let mg_factor = phase;
        let mut occ_opp = bitboards.occupied(opponent);
        while occ_opp != 0 {
            let sq = Bitboards::lsb_index(occ_opp);
            if let Some(p) = bitboards.piece_at(sq) {
                let x = sq % 9;
                let y = sq / 9;
                let pos = Coord::new(x as i8, y as i8);
                let dist = (pos.x - king_pos.x).abs() + (pos.y - king_pos.y).abs();
                let threat = match p.piece_type {
                    PieceType::Chariot => (14 - dist).max(0) as i32 * 10,
                    PieceType::Cannon => (12 - dist).max(0) as i32 * 7,
                    PieceType::Horse => (10 - dist).max(0) as i32 * 8,
                    PieceType::Pawn if dist <= 4 => (5 - dist) as i32 * 8,
                    _ => 0,
                };
                safety -= threat * mg_factor / TOTAL_PHASE;
            }
            occ_opp &= occ_opp - 1;
        }

        // 2. Attack pressure near enemy king — our pieces near their king
        let enemy_king_pos = match color {
            Color::Red => bk,
            Color::Black => rk,
        };

        if let Some(ek) = enemy_king_pos {
            let mut occ_our = bitboards.occupied(color);
            while occ_our != 0 {
                let sq = Bitboards::lsb_index(occ_our);
                if let Some(p) = bitboards.piece_at(sq) {
                    let x = sq % 9;
                    let y = sq / 9;
                    let our_pos = Coord::new(x as i8, y as i8);
                    let dist = our_pos.distance_to(ek);
                    if dist <= 2 {
                        let pressure = match p.piece_type {
                            PieceType::Chariot | PieceType::Cannon => 15,
                            PieceType::Horse => 10,
                            PieceType::Pawn if our_pos.crosses_river(color) => 5,
                            _ => 0,
                        };
                        safety += pressure;
                    }
                }
                occ_our &= occ_our - 1;
            }
        }

        Some(safety * color.sign())
    }

    fn chariot_mobility(board: &Board, pos: Coord, _color: Color) -> i32 {
        // Chariot mobility: empty squares in open lines × 5
        let mut score = 0;

        for (dx, dy) in DIRS_4 {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;

            while (0..BOARD_WIDTH).contains(&x) && (0..BOARD_HEIGHT).contains(&y) {
                let tar = Coord::new(x, y);
                if board.get(tar).is_some() {
                    break; // Blocked, no score for this direction
                }
                score += 5; // Each empty square is worth 5
                x += dx;
                y += dy;
            }
        }
        score
    }

    fn pawn_structure(board: &Board, color: Color, phase: i32) -> i32 {
        use crate::Bitboards;
        let mut score = 0;
        let eg_factor = TOTAL_PHASE - phase;
        let bitboards = &board.bitboards;

        // Count pawns per file for doubled pawn detection
        let mut pawns_per_file = [0i32; 9];

        let mut occ = bitboards.occupied(color);
        while occ != 0 {
            let sq = Bitboards::lsb_index(occ);
            if let Some(p) = bitboards.piece_at(sq) {
                if p.piece_type == PieceType::Pawn {
                    let x = sq % 9;
                    let y = sq / 9;
                    pawns_per_file[x as usize] += 1;

                    let file_count = pawns_per_file[x as usize];
                    // Doubled pawn penalty: -30 per pawn on a file with 2+ pawns
                    if file_count >= 2 {
                        score -= 30;
                    }

                    // Back-rank pawn penalty: -20 (not crossed river)
                    let not_crossed = match color {
                        Color::Red => y == 6,
                        Color::Black => y == 3,
                    };
                    if not_crossed {
                        score -= 20;
                    }

                    // Pawn linkage: check left and right neighbors on same rank
                    let left_sq = sq - 1;
                    let right_sq = sq + 1;
                    let mut linked = 0;
                    if left_sq / 9 == y {
                        if let Some(lp) = bitboards.piece_at(left_sq) {
                            if lp.piece_type == PieceType::Pawn && lp.color == color {
                                linked += 1;
                            }
                        }
                    }
                    if right_sq / 9 == y {
                        if let Some(rp) = bitboards.piece_at(right_sq) {
                            if rp.piece_type == PieceType::Pawn && rp.color == color {
                                linked += 1;
                            }
                        }
                    }
                    score += linked * (20 * TOTAL_PHASE / 82 + 30 * eg_factor / 82);

                    // Passed pawn bonus in endgame
                    let y_i8 = y as i8;
                    let crossed = match color {
                        Color::Red => y_i8 <= 4,
                        Color::Black => y_i8 >= 5,
                    };
                    if eg_factor > TOTAL_PHASE / 2 && crossed {
                        let (rk, bk) = board.find_kings();
                        let enemy_king = if color == Color::Red { bk } else { rk };
                        if let Some(ek) = enemy_king {
                            let dist = ((x as i8) - ek.x).abs() + ((y as i8) - ek.y).abs();
                            if dist <= 2 {
                                score += (3 - dist) as i32 * 50;
                            }
                        }
                    }

                    // Advancement bonus
                    let crossed = match color {
                        Color::Red => y_i8 <= 4,
                        Color::Black => y_i8 >= 5,
                    };
                    if crossed {
                        score += match color {
                            Color::Red => {
                                if y == 0 { 80 }
                                else if y <= 3 { 5 }
                                else { 0 }
                            },
                            Color::Black => {
                                if y == 9 { 80 }
                                else if y >= 6 { 5 }
                                else { 0 }
                            }
                        };
                    }

                    // Central file bonus: pawn on file 4 or 5 gets bonus per rank advanced
                    if x == 4 || x == 5 {
                        let ranks_from_origin: i32 = match color {
                            Color::Red => (6 - y as i8) as i32,
                            Color::Black => (y as i8 - 3) as i32,
                        };
                        if ranks_from_origin > 0 {
                            score += ranks_from_origin * 10;
                        }
                    }
                }
            }
            occ &= occ - 1;
        }
        score * color.sign()
    }

    fn elephant_structure(board: &Board, color: Color, phase: i32) -> i32 {
        use crate::Bitboards;
        let mut score = 0;
        let bitboards = &board.bitboards;

        // At most 2 elephants per side
        let mut elephants = SmallVec::<[Coord; 2]>::new();

        let mut occ = bitboards.occupied(color);
        while occ != 0 {
            let sq = Bitboards::lsb_index(occ);
            if let Some(p) = bitboards.piece_at(sq) {
                if p.piece_type == PieceType::Elephant {
                    let x = sq % 9;
                    let y = sq / 9;
                    elephants.push(Coord::new(x as i8, y as i8));
                }
            }
            occ &= occ - 1;
        }

        let count = elephants.len();
        let mg_factor = phase;
        let eg_factor = TOTAL_PHASE - phase;

        match count {
            0 => {
                // Missing both elephants: penalty ≈ full elephant value (80 MG / 200 EG)
                score -= (80 * mg_factor + 200 * eg_factor) / TOTAL_PHASE;
            }
            1 => {
                // Missing one elephant: penalty ≈ half elephant value
                score -= (40 * mg_factor + 100 * eg_factor) / TOTAL_PHASE;
                if let Some(pos) = elephants.first() {
                    let moves = movegen::generate_elephant_moves(board, *pos, color);
                    let mut protected = false;
                    for m in moves {
                        if let Some(p) = board.get(m)
                            && p.color == color && (p.piece_type == PieceType::Advisor || p.piece_type == PieceType::Pawn) {
                                protected = true;
                                break;
                            }
                    }
                    if !protected {
                        score -= 15;
                    }
                }
            }
            2 => {
                let pos1 = elephants[0];
                let pos2 = elephants[1];

                // Generate moves for both elephants
                let moves1 = movegen::generate_elephant_moves(board, pos1, color);
                let moves2 = movegen::generate_elephant_moves(board, pos2, color);

                // Check mutual protection: they share defensive positions
                // Each elephant can move to squares the other can reach
                let mut mutual_protection = false;
                for m1 in &moves1 {
                    if moves2.contains(m1) {
                        mutual_protection = true;
                        break;
                    }
                }

                if mutual_protection {
                    score += (40 * mg_factor + 30 * eg_factor) / TOTAL_PHASE;
                } else {
                    // Even without direct mutual protection, having 2 elephants is still a structure bonus
                    score += (20 * mg_factor + 15 * eg_factor) / TOTAL_PHASE;
                }
            }
            _ => {}
        }

        score * color.sign()
    }

    pub fn handcrafted_evaluate(board: &Board, side: Color, initiative: bool) -> i32 {
        // Compute regular evaluation result first (needed for blending)
        let (rk, bk) = board.find_kings();

        // Handle checkmate positions
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
        let mut score = 0;

        use crate::Bitboards;
        let bitboards = &board.bitboards;
        let mut occ = bitboards.occupied_all();
        while occ != 0 {
            let sq = Bitboards::lsb_index(occ);
            if let Some(piece) = bitboards.piece_at(sq) {
                let x = sq % 9;
                let y = sq / 9;
                let x_usize = x as usize;
                let y_usize = y as usize;
                let sign = piece.color.sign();
                let pos = Coord::new(x as i8, y as i8);

                let mg_v = MG_VALUE[piece.piece_type as usize];
                let eg_v = EG_VALUE[piece.piece_type as usize];
                let val = (mg_v * phase + eg_v * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
                score += val * sign;

                let pst = pst_val(piece.piece_type, piece.color, x_usize, y_usize, phase);
                score += pst;

                match piece.piece_type {
                    PieceType::Horse => {
                        let mob = horse_mobility(board, pos, piece.color);
                        score += mob * sign;
                        let near_home = match piece.color {
                            Color::Red => pos.y >= 5 && pos.y <= 6,
                            Color::Black => pos.y >= 3 && pos.y <= 4,
                        };
                        if near_home {
                            score += 15 * sign;
                        }
                    }
                    PieceType::Chariot => {
                        let mob = chariot_mobility(board, pos, piece.color);
                        score += mob * sign;
                        let on_enemy_back_rank = match piece.color {
                            Color::Red => pos.y <= 1,
                            Color::Black => pos.y >= 8,
                        };
                        if on_enemy_back_rank {
                            score += 40 * sign;
                        }
                    }
                    PieceType::Cannon => {
                        let sup = cannon_support(board, pos, piece.color);
                        score += sup * sign;
                    }
                    _ => {}
                }
            }
            occ &= occ - 1;
        }

        if let Some(ks_red) = king_safety(board, Color::Red, phase) {
            score += ks_red;
        }
        if let Some(ks_black) = king_safety(board, Color::Black, phase) {
            score += ks_black;
        }

        // King attack bonus: pieces directly threatening enemy king get +25 each
        let (rk, bk) = board.find_kings();
        for (king_color, enemy_king_pos) in [(Color::Red, bk), (Color::Black, rk)] {
            if let Some(kp) = enemy_king_pos {
                // Count how many pieces of the OTHER color (the attacker) can attack this king
                let attacker_color = king_color.opponent();
                let mut attack_count = 0;
                let mut occ_att = bitboards.occupied(attacker_color);
                while occ_att != 0 {
                    let sq = Bitboards::lsb_index(occ_att);
                    if let Some(_p) = bitboards.piece_at(sq) {
                        let x = sq % 9;
                        let y = sq / 9;
                        let pos = Coord::new(x as i8, y as i8);
                        let dist = pos.distance_to(kp);
                        if dist == 1 {
                            attack_count += 1;
                        }
                    }
                    occ_att &= occ_att - 1;
                }
                let bonus = attack_count * 25 * king_color.sign();
                score += bonus;
            }
        }

        score += pawn_structure(board, Color::Red, phase);
        score += pawn_structure(board, Color::Black, phase);

        score += center_control(board, Color::Red, phase);
        score += center_control(board, Color::Black, phase);
        score += piece_coordination(board, Color::Red, phase);
        score += piece_coordination(board, Color::Black, phase);

        score += elephant_structure(board, Color::Red, phase);
        score += elephant_structure(board, Color::Black, phase);

        // Attack rewards
        score += attack_rewards(board, Color::Red, phase);
        score += attack_rewards(board, Color::Black, phase);

        if board.is_check(Color::Black) {
            score += CHECK_BONUS;
        }
        if board.is_check(Color::Red) {
            score -= CHECK_BONUS;
        }

        if initiative {
            score += 20 * side.sign();
        }

        let regular_score = score * side.sign();

        // Blend with tablebase if available
        if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
            if conf < 0.2 { return tb_score; }
            let weight = conf.max(0.3);
            return (tb_score as f32 * weight + regular_score as f32 * (1.0 - weight)) as i32;
        }

        regular_score
    }
}
