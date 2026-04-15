//! Bitboard representation for Xiangqi (Chinese Chess).
//!
//! Uses u128 per piece type per color (14 total bitboards for 7 piece types × 2 colors).
//! Board is 9×10 = 90 squares, bits 90-127 unused (always 0).
//!
//! Bit index formula: `sq = y * 9 + x` where x=0-8, y=0-9

#[allow(dead_code)]
use crate::{Color, Piece, PieceType, Coord};
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Number of squares on the Xiangqi board
pub const BOARD_SQ_COUNT: usize = 90;

/// Convert board Coord to bitboard square index (0-89)
#[inline(always)]
#[allow(dead_code)]
pub fn sq_from_coord(x: i8, y: i8) -> u8 {
    (y * 9 + x) as u8
}

/// Convert bitboard square index (0-89) to Coord
#[inline(always)]
#[allow(dead_code)]
pub fn coord_from_sq(sq: u8) -> Coord {
    Coord::new((sq % 9) as i8, (sq / 9) as i8)
}

/// Returns true if the square index is within the 90-board range
#[inline(always)]
pub fn is_valid_sq(sq: u8) -> bool {
    sq < BOARD_SQ_COUNT as u8
}

/// CHARIOT_RAYS_STORAGE[sq][dir] = u128 mask of all squares from sq in cardinal direction.
/// dir: 0=North(+y), 1=South(-y), 2=East(+x), 3=West(-x)
static CHARIOT_RAYS_STORAGE: OnceLock<[[u128; 4]; BOARD_SQ_COUNT]> = OnceLock::new();
/// CANNON_SCREENS_STORAGE[sq][dir] = mask of squares between sq and first blocker (used for cannon captures)
#[allow(dead_code)]
static CANNON_SCREENS_STORAGE: OnceLock<[[u128; 4]; BOARD_SQ_COUNT]> = OnceLock::new();

/// Initialize CHARIOT_RAYS and CANNON_SCREENS tables.
/// Each entry is a u128 mask of squares along a ray from sq in one direction.
fn init_chariot_rays() -> ([[u128; 4]; BOARD_SQ_COUNT], [[u128; 4]; BOARD_SQ_COUNT]) {
    let mut chariot_rays = [[0u128; 4]; BOARD_SQ_COUNT];
    let mut cannon_screens = [[0u128; 4]; BOARD_SQ_COUNT];

    for sq in 0..BOARD_SQ_COUNT {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;

        // Direction 0: North (+y, toward y=9)
        // Sets bits near→far: 54,63,72,81 → lsb_index = nearest (54)
        let mut mask = 0u128;
        let mut ny = y + 1;
        while ny < 10 {
            let nsq = (ny * 9 + x) as u8;
            mask |= 1_u128 << nsq;
            cannon_screens[sq][0] |= mask & !(1_u128 << nsq);
            ny += 1;
        }
        chariot_rays[sq][0] = mask;

        // Direction 1: South (-y, toward y=0)
        // Sets bits far→near: 36,27,18,9,0 → lsb_index = furthest (0), msb_index = nearest (36)
        mask = 0u128;
        let mut ny = y - 1;
        while ny >= 0 {
            let nsq = (ny * 9 + x) as u8;
            mask |= 1_u128 << nsq;
            cannon_screens[sq][1] |= mask & !(1_u128 << nsq);
            ny -= 1;
        }
        chariot_rays[sq][1] = mask;

        // Direction 2: East (+x, toward x=8)
        // Sets bits near→far: 46,47,48 → lsb_index = nearest (46)
        mask = 0u128;
        let mut nx = x + 1;
        while nx < 9 {
            let nsq = (y * 9 + nx) as u8;
            mask |= 1_u128 << nsq;
            cannon_screens[sq][2] |= mask & !(1_u128 << nsq);
            nx += 1;
        }
        chariot_rays[sq][2] = mask;

        // Direction 3: West (-x, toward x=0)
        // Sets bits far→near: 44,43,42... → lsb_index = furthest, msb_index = nearest
        mask = 0u128;
        let mut nx = x - 1;
        while nx >= 0 {
            let nsq = (y * 9 + nx) as u8;
            mask |= 1_u128 << nsq;
            cannon_screens[sq][3] |= mask & !(1_u128 << nsq);
            nx -= 1;
        }
        chariot_rays[sq][3] = mask;
    }

    (chariot_rays, cannon_screens)
}

/// Get CHARIOT_RAYS table (lazily initialized)
pub fn get_chariot_rays() -> &'static [[u128; 4]; BOARD_SQ_COUNT] {
    CHARIOT_RAYS_STORAGE.get_or_init(|| init_chariot_rays().0)
}

/// Get CANNON_SCREENS table (lazily initialized)
#[allow(dead_code)]
pub fn get_cannon_screens() -> &'static [[u128; 4]; BOARD_SQ_COUNT] {
    CANNON_SCREENS_STORAGE.get_or_init(|| init_chariot_rays().1)
}

/// Bitboards for a Xiangqi position.
/// pieces[piece_type][color] → u128 bitboard
/// piece_type index: 0=King, 1=Advisor, 2=Elephant, 3=Pawn, 4=Horse, 5=Cannon, 6=Chariot
/// color index: 0=Red, 1=Black
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboards {
    pieces: [[u128; 2]; 7],
}

impl Bitboards {
    /// Create empty bitboards (no pieces)
    pub fn new() -> Self {
        Bitboards { pieces: [[0u128; 2]; 7] }
    }

    /// Build bitboards from the existing cells array
    pub fn from_cells(cells: &[[Option<Piece>; 9]; 10]) -> Self {
        let mut bb = Bitboards::new();
        for (y, row) in cells.iter().enumerate().take(10) {
            for (x, &opt_piece) in row.iter().enumerate().take(9) {
                if let Some(piece) = opt_piece {
                    let sq = (y * 9 + x) as u8;
                    bb.pieces[piece.piece_type as usize][piece.color as usize] |= 1_u128 << sq;
                }
            }
        }
        bb
    }

    /// All occupied squares for a given color (union of all its piece bitboards)
    #[inline(always)]
    pub fn occupied(&self, color: Color) -> u128 {
        let c = color as usize;
        self.pieces[0][c] | self.pieces[1][c] | self.pieces[2][c]
            | self.pieces[3][c] | self.pieces[4][c] | self.pieces[5][c]
            | self.pieces[6][c]
    }

    /// All occupied squares on the board (both colors)
    #[inline(always)]
    pub fn occupied_all(&self) -> u128 {
        self.occupied(Color::Red) | self.occupied(Color::Black)
    }

    /// Get the bitboard for a specific piece type and color.
    #[inline(always)]
    pub fn piece_bitboard(&self, piece_type: PieceType, color: Color) -> u128 {
        self.pieces[piece_type as usize][color as usize]
    }

    /// Returns the piece at the given square, or None if empty.
    /// Returns None for invalid squares (outside 0-89).
    pub fn piece_at(&self, sq: u8) -> Option<Piece> {
        if !is_valid_sq(sq) {
            return None;
        }
        let bb = 1_u128 << sq;
        // Search all piece types and colors
        for pt in 0..7 {
            for c in 0..2 {
                if self.pieces[pt][c] & bb != 0 {
                    // Safety: pt is always 0-6 (valid PieceType discriminants) due to loop bounds
                    debug_assert!(pt < 7, "Invalid piece type index {}", pt);
                    return Some(Piece {
                        color: if c == 0 { Color::Red } else { Color::Black },
                        piece_type: unsafe { std::mem::transmute::<u8, PieceType>(pt as u8) },
                    });
                }
            }
        }
        None
    }

    /// Returns the piece at the given Coord, or None if empty/invalid.
    pub fn piece_at_coord(&self, coord: Coord) -> Option<Piece> {
        let sq = (coord.y * 9 + coord.x) as u8;
        self.piece_at(sq)
    }

    /// Reconstruct the 10x9 cells array from bitboards.
    /// Used primarily for testing to maintain compatibility with tests that
    /// reference board cells.
    pub fn as_cells(&self) -> [[Option<Piece>; 9]; 10] {
        let mut cells = [[None; 9]; 10];
        for (y, row) in cells.iter_mut().enumerate().take(10) {
            for (x, cell) in row.iter_mut().enumerate().take(9) {
                *cell = self.piece_at((y * 9 + x) as u8);
            }
        }
        cells
    }

    /// Flip the bitboards vertically (swap rows 0↔9, 1↔8, 2↔7, 3↔6, 4↔5).
    /// This is used for display purposes when showing the board from Black's perspective.
    pub fn flip_vertically(&mut self) {
        let mut new_pieces = [[0u128; 2]; 7];

        for (pt, new_piece_row) in new_pieces.iter_mut().enumerate() {
            for (c, new_bb_slot) in new_piece_row.iter_mut().enumerate() {
                let bb = self.pieces[pt][c];
                let mut new_bb = 0u128;
                let mut sq = 0u8;
                while sq < 90 {
                    if bb & (1_u128 << sq) != 0 {
                        let y = sq / 9;
                        let x = sq % 9;
                        let new_sq = (9 - y) * 9 + x;
                        new_bb |= 1_u128 << new_sq;
                    }
                    sq += 1;
                }
                *new_bb_slot = new_bb;
            }
        }

        self.pieces = new_pieces;
    }

    /// Find the index of the least significant set bit in a u128.
    /// Used to find the nearest blocker along a ray.
    #[inline(always)]
    pub fn lsb_index(bb: u128) -> u8 {
        bb.trailing_zeros() as u8
    }

    /// Find the index of the most significant set bit in a u128.
    #[inline(always)]
    pub fn msb_index(bb: u128) -> u8 {
        127 - bb.leading_zeros() as u8
    }

    /// Chariot attacks from sq — all squares in 4 cardinal directions until first blocker.
    /// Returns all reachable squares (empty OR enemy-capturable). Friendly pieces block.
    pub fn chariot_attacks(&self, sq: u8, color: Color) -> u128 {
        let occ = self.occupied_all();
        let occ_color = self.occupied(color);
        let rays = get_chariot_rays();
        let mut attacks = 0u128;

        for (dir, &ray) in rays[sq as usize].iter().enumerate().take(4) {
            let blockers = ray & occ;
            if blockers == 0 {
                attacks |= ray;  // Clear path — all squares reachable
            } else {
                // FIX: Nearest piece depends on direction:
                // - North/East: nearest = smallest square = lsb_index
                // - South/West: nearest = largest square = msb_index
                let nearest = if dir % 2 == 0 {
                    Self::lsb_index(blockers)  // North or East
                } else {
                    Self::msb_index(blockers)   // South or West
                };
                // Only include squares up to nearest if it is NOT our piece
                if occ_color & (1_u128 << nearest) == 0 {
                    let ray_to_nearest = ray & !(rays[nearest as usize][dir]);
                    attacks |= ray_to_nearest;  // Includes nearest enemy square (capture)
                }
                // If nearest is our own piece: no squares attacked in this direction (blocked)
            }
        }
        attacks
    }

    /// Cannon attacks from sq — slides until first screen, then captures through it.
    /// Filters out own-occupied squares from capture destinations.
    pub fn cannon_attacks(&self, sq: u8, color: Color) -> u128 {
        let occ = self.occupied_all();
        let occ_color = self.occupied(color);
        let rays = get_chariot_rays();
        let mut attacks = 0u128;

        for (dir, &ray) in rays[sq as usize].iter().enumerate().take(4) {
            let blockers = ray & occ;
            if blockers == 0 {
                // No screen, no captures - all squares along ray are empty (already in ray)
                attacks |= ray;
            } else {
                // FIX: Nearest screen depends on direction:
                // - North/East: nearest = smallest square = lsb_index
                // - South/West: nearest = largest square = msb_index
                let nearest = if dir % 2 == 0 {
                    Self::lsb_index(blockers)  // North or East
                } else {
                    Self::msb_index(blockers)   // South or West
                };
                // Quiet moves: squares BEFORE the screen (empty squares only, not including screen or beyond)
                // rays[nearest] = squares BEYOND nearest
                // Also exclude occupied squares (both own and enemy)
                let quiet_ray = ray & !(rays[nearest as usize][dir]) & !occ;
                attacks |= quiet_ray;

                // Capture: squares BEYOND the screen that are occupied by enemy
                // (blockers in the ray beyond the nearest screen)
                let second_blockers = ray & occ & rays[nearest as usize][dir];
                if second_blockers != 0 {
                    // Second blocker: same direction logic (it's the nearest remaining blocker)
                    let second = if dir % 2 == 0 {
                        Self::lsb_index(second_blockers)
                    } else {
                        Self::msb_index(second_blockers)
                    };
                    // Capture only if target is NOT our own piece
                    if occ_color & (1_u128 << second) == 0 {
                        attacks |= 1_u128 << second;
                    }
                }
            }
        }
        attacks
    }

    /// Horse attacks from sq — 8 L-shape destinations, knee square must be empty.
    /// Filters out own-occupied destination squares.
    pub fn horse_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let occ = self.occupied_all();
        let occ_color = self.occupied(color);

        let horse_deltas: [(i8, i8); 8] = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ];
        let horse_blocks: [(i8, i8); 8] = [
            (1, 0), (1, 0), (-1, 0), (-1, 0),
            (0, 1), (0, -1), (0, 1), (0, -1)
        ];

        let mut attacks = 0u128;
        for i in 0..8 {
            let (dx, dy) = horse_deltas[i];
            let (bx, by) = horse_blocks[i];
            let tar_x = x + dx;
            let tar_y = y + dy;
            let knee_x = x + bx;
            let knee_y = y + by;

            // Knee must be on board and empty
            if !(0..9).contains(&knee_x) || !(0..10).contains(&knee_y) {
                continue;
            }
            let knee_sq = (knee_y * 9 + knee_x) as u8;
            if occ & (1_u128 << knee_sq) != 0 {
                continue; // knee is blocked
            }

            // Target must be on board
            if !(0..9).contains(&tar_x) || !(0..10).contains(&tar_y) {
                continue;
            }

            let tar_sq = (tar_y * 9 + tar_x) as u8;
            // Target must not be own-occupied
            if occ_color & (1_u128 << tar_sq) != 0 {
                continue;
            }

            attacks |= 1_u128 << tar_sq;
        }
        attacks
    }

    /// Advisor attacks from sq — 4 diagonal destinations, palace-bound.
    /// Filters out own-occupied destination squares.
    pub fn advisor_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let occ_color = self.occupied(color);

        let deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut attacks = 0u128;
        for (dx, dy) in deltas {
            let tx = x + dx;
            let ty = y + dy;
            let target = Coord::new(tx, ty);
            // Palace bounds check
            if !target.is_valid() || !target.in_palace(color) {
                continue;
            }
            let tsq = (ty * 9 + tx) as u8;
            if occ_color & (1_u128 << tsq) == 0 {
                attacks |= 1_u128 << tsq;
            }
        }
        attacks
    }

    /// Elephant attacks from sq — 4 diagonal destinations, eye must be empty, cannot cross river.
    /// Filters out own-occupied destination squares.
    pub fn elephant_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let occ_color = self.occupied(color);
        let occ_all = self.occupied_all();

        let deltas = [(2, 2), (2, -2), (-2, 2), (-2, -2)];
        let blocks = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut attacks = 0u128;
        for i in 0..4 {
            let (dx, dy) = deltas[i];
            let (bx, by) = blocks[i];
            let tx = x + dx;
            let ty = y + dy;
            let bx = x + bx;
            let by = y + by;

            // Eye square must be empty
            if (0..9).contains(&bx) && (0..10).contains(&by) {
                let eye_sq = (by * 9 + bx) as u8;
                if occ_all & (1_u128 << eye_sq) != 0 {
                    continue;
                }
            } else {
                continue;
            }

            // Target must be on board
            if !(0..9).contains(&tx) || !(0..10).contains(&ty) {
                continue;
            }

            // River check: elephant cannot cross
            let target_coord = Coord::new(tx, ty);
            if target_coord.crosses_river(color) {
                continue;
            }

            let tsq = (ty * 9 + tx) as u8;
            // Filter own-occupied
            if occ_color & (1_u128 << tsq) == 0 {
                attacks |= 1_u128 << tsq;
            }
        }
        attacks
    }

    /// King attacks from sq — 4 orthogonal destinations, palace-bound.
    /// Filters out own-occupied destination squares.
    pub fn king_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let occ_color = self.occupied(color);

        let offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut attacks = 0u128;
        for (dx, dy) in offsets {
            let tx = x + dx;
            let ty = y + dy;
            let target = Coord::new(tx, ty);
            // Palace bounds check
            if !target.is_valid() || !target.in_palace(color) {
                continue;
            }
            let tsq = (ty * 9 + tx) as u8;
            if occ_color & (1_u128 << tsq) == 0 {
                attacks |= 1_u128 << tsq;
            }
        }
        attacks
    }

    /// Pawn attacks from sq for the given color.
    /// Returns attack DESTINATION squares — where this pawn can attack.
    /// Filters out own-occupied destination squares.
    pub fn pawn_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let dir: i8 = if color == Color::Red { -1 } else { 1 };
        let occ_color = self.occupied(color);
        let mut attacks = 0u128;

        let forward_sq = (y + dir, x);
        if forward_sq.0 >= 0 && forward_sq.0 < 10 {
            let fsq = (forward_sq.0 * 9 + forward_sq.1) as u8;
            if occ_color & (1_u128 << fsq) == 0 {
                attacks |= 1_u128 << fsq;
            }
        }

        // Side moves only after crossing river
        let crossed = if color == Color::Red { y <= 4 } else { y >= 5 };
        if crossed {
            for dx in [-1, 1] {
                let sx = x + dx;
                if (0..9).contains(&sx) {
                    let ssq = (y * 9 + sx) as u8;
                    if occ_color & (1_u128 << ssq) == 0 {
                        attacks |= 1_u128 << ssq;
                    }
                }
            }
        }
        attacks
    }

    /// Returns a u128 bitboard of all squares containing pieces of `color`
    /// that can attack the given `target` square.
    /// This is the core building block for SEE.
    pub fn attackers(&self, target: u8, color: Color) -> u128 {
        let occ = self.occupied_all();
        let occ_color = self.occupied(color);
        let rays = get_chariot_rays();
        let mut attackers = 0u128;

        // Chariot attacks: slides in 4 directions, nearest piece in each direction
        for (dir, ray) in rays[target as usize].iter().enumerate().take(4) {
            let blockers = ray & occ;
            if blockers == 0 { continue; }
            // FIX: Nearest depends on direction: North/East → lsb, South/West → msb
            let nearest = if dir % 2 == 0 {
                Self::lsb_index(blockers)  // North or East
            } else {
                Self::msb_index(blockers)   // South or West
            };
            if occ_color & (1_u128 << nearest) != 0
                && self.pieces[PieceType::Chariot as usize][color as usize] & (1_u128 << nearest) != 0
            {
                attackers |= 1_u128 << nearest;
            }
        }

        // Cannon attacks: needs exactly 1 screen between src and target
        // Screen can be ANY color; cannon must be attacker's piece
        for (dir, &ray) in rays[target as usize].iter().enumerate().take(4) {
            let blockers = ray & occ;
            if blockers == 0 { continue; }
            // Nearest screen depends on direction
            let nearest = if dir % 2 == 0 {
                Self::lsb_index(blockers)  // North or East
            } else {
                Self::msb_index(blockers)   // South or West
            };
            // Screen can be any color - find second blocker beyond screen
            // rays[nearest][dir] = squares from screen going in same direction (beyond screen)
            let second_blockers = ray & occ & rays[nearest as usize][dir];
            if second_blockers != 0 {
                // Second blocker must be attacker's cannon
                let cannon_mask = second_blockers & self.pieces[PieceType::Cannon as usize][color as usize];
                if cannon_mask != 0 {
                    // Target must be enemy-occupied
                    if self.piece_at(target).map_or(false, |p| p.color != color) {
                        attackers |= cannon_mask;
                    }
                }
            }
        }

        // Horse attacks: 8 L-shape destinations around target
        // Horse at SRC attacks TAR: SRC = TAR - HORSE_DELTA
        let horse_attacks_bb = self.horse_attacks(target, color);
        let mut horse_bb = horse_attacks_bb & occ_color;
        while horse_bb != 0 {
            let sq = Self::lsb_index(horse_bb);
            if self.pieces[PieceType::Horse as usize][color as usize] & (1_u128 << sq) != 0 {
                attackers |= 1_u128 << sq;
            }
            horse_bb &= horse_bb - 1;
        }

        // Pawn attacks: find SOURCE positions where pawns can attack target
        // For target at (tx, ty), pawn must be at:
        // - Forward: (tx, ty - dir) = (ty - dir, tx) in code's (y, x) ordering
        // - Side: (tx±1, ty) = (ty, tx±1) in code's (y, x) ordering
        let tx = (target % 9) as i8;
        let ty = (target / 9) as i8;
        let dir: i8 = if color == Color::Red { -1 } else { 1 };

        // Forward SOURCE: (ty - dir, tx)
        let forward_y = ty - dir;
        if forward_y >= 0 && forward_y < 10 {
            let forward_sq = (forward_y * 9 + tx) as u8;
            if occ_color & (1_u128 << forward_sq) != 0
                && self.pieces[PieceType::Pawn as usize][color as usize] & (1_u128 << forward_sq) != 0
            {
                attackers |= 1_u128 << forward_sq;
            }
        }

        // Side SOURCE: (ty, tx±1)
        // Side attacks only after crossing river
        let crossed = if color == Color::Red { ty <= 4 } else { ty >= 5 };
        if crossed {
            for dx in [-1, 1] {
                let sx = tx + dx;
                if (0..9).contains(&sx) {
                    let side_sq = (ty * 9 + sx) as u8;
                    if occ_color & (1_u128 << side_sq) != 0
                        && self.pieces[PieceType::Pawn as usize][color as usize] & (1_u128 << side_sq) != 0
                    {
                        attackers |= 1_u128 << side_sq;
                    }
                }
            }
        }

        // King attacks: 4 orthogonal moves
        let king_attacks_bb = self.king_attacks(target, color);
        let mut king_bb = king_attacks_bb & occ_color;
        while king_bb != 0 {
            let sq = Self::lsb_index(king_bb);
            if self.pieces[PieceType::King as usize][color as usize] & (1_u128 << sq) != 0 {
                attackers |= 1_u128 << sq;
            }
            king_bb &= king_bb - 1;
        }

        // Advisor attacks: iterate through all advisor squares and check if they attack target
        let advisor_bb = self.pieces[PieceType::Advisor as usize][color as usize];
        let mut tmp = advisor_bb;
        while tmp != 0 {
            let sq = Self::lsb_index(tmp);
            let attacks_from_sq = self.advisor_attacks(sq, color);
            if attacks_from_sq & (1_u128 << target) != 0 {
                attackers |= 1_u128 << sq;
            }
            tmp &= tmp - 1;
        }

        // Elephant attacks: iterate through all elephant squares and check if they attack target
        let elephant_bb = self.pieces[PieceType::Elephant as usize][color as usize];
        let mut tmp = elephant_bb;
        while tmp != 0 {
            let sq = Self::lsb_index(tmp);
            let attacks_from_sq = self.elephant_attacks(sq, color);
            if attacks_from_sq & (1_u128 << target) != 0 {
                attackers |= 1_u128 << sq;
            }
            tmp &= tmp - 1;
        }

        attackers
    }

    /// Generate all pseudo-legal move destination squares for a piece at `from`.
    /// Returns a Vec of destination squares (u8 bitboard indices).
    /// Caller filters by occupancy and color to determine captures vs quiet moves.
    pub fn generate_moves(&self, from: u8, color: Color) -> SmallVec<[u8; 17]> {
        let mut destinations = SmallVec::new();
        let attacks = match () {
            _ if self.pieces[PieceType::Chariot as usize][color as usize] & (1_u128 << from) != 0 => {
                self.chariot_attacks(from, color)
            }
            _ if self.pieces[PieceType::Cannon as usize][color as usize] & (1_u128 << from) != 0 => {
                self.cannon_attacks(from, color)
            }
            _ if self.pieces[PieceType::Horse as usize][color as usize] & (1_u128 << from) != 0 => {
                self.horse_attacks(from, color)
            }
            _ if self.pieces[PieceType::Advisor as usize][color as usize] & (1_u128 << from) != 0 => {
                self.advisor_attacks(from, color)
            }
            _ if self.pieces[PieceType::Elephant as usize][color as usize] & (1_u128 << from) != 0 => {
                self.elephant_attacks(from, color)
            }
            _ if self.pieces[PieceType::King as usize][color as usize] & (1_u128 << from) != 0 => {
                self.king_attacks(from, color)
            }
            _ if self.pieces[PieceType::Pawn as usize][color as usize] & (1_u128 << from) != 0 => {
                self.pawn_attacks(from, color)
            }
            _ => 0u128,
        };

        let mut bb = attacks;
        while bb != 0 {
            let dst = Self::lsb_index(bb);
            destinations.push(dst);
            bb &= bb - 1; // Clear LSB
        }
        destinations
    }

    /// Generate all pseudo-legal moves for a color (piece at each occupied square).
    /// Returns vector of (src, dst, captured_piece) tuples.
    pub fn generate_pseudo_moves(&self, color: Color) -> SmallVec<[(u8, u8, Option<Piece>); 64]> {
        let mut moves = SmallVec::new();
        let own = self.occupied(color);

        let mut bb = own;
        while bb != 0 {
            let from = Self::lsb_index(bb);
            let dsts = self.generate_moves(from, color);
            for dst in dsts {
                let captured = self.piece_at(dst);
                moves.push((from, dst, captured));
            }
            bb &= bb - 1; // Clear LSB
        }
        moves
    }

    /// Apply a move to the bitboards in place.
    /// `src` and `dst` are bitboard square indices (0-89).
    /// `captured` is the piece that was on dst before the move (None if empty).
    /// `piece` is the piece being moved.
    pub fn apply_move(&mut self, src: u8, dst: u8, captured: Option<Piece>, piece: Piece) {
        // Remove from src
        self.pieces[piece.piece_type as usize][piece.color as usize] &= !(1_u128 << src);
        // Add to dst
        self.pieces[piece.piece_type as usize][piece.color as usize] |= 1_u128 << dst;
        // Remove captured piece
        if let Some(cp) = captured {
            self.pieces[cp.piece_type as usize][cp.color as usize] &= !(1_u128 << dst);
        }
    }

    /// Undo a move — restores piece to src and captured piece to dst.
    pub fn undo_move(&mut self, src: u8, dst: u8, captured: Option<Piece>, piece: Piece) {
        // Remove from dst
        self.pieces[piece.piece_type as usize][piece.color as usize] &= !(1_u128 << dst);
        // Restore to src
        self.pieces[piece.piece_type as usize][piece.color as usize] |= 1_u128 << src;
        // Restore captured piece
        if let Some(cp) = captured {
            self.pieces[cp.piece_type as usize][cp.color as usize] |= 1_u128 << dst;
        }
    }

    /// Fill the NNUE input feature planes from bitboards (side-to-move perspective).
    /// Populates stm_data[1260] array where stm squares are at base 0, opponent at base 630.
    /// This matches the encoding produced by NNInputPlanes::from_board().
    pub fn fill_nnue_features(&self, stm: Color, stm_data: &mut [f32; 1260]) {
        let _ntm = stm.opponent();

        for pt in 0..7 {
            for c in 0..2 {
                let bb = self.pieces[pt][c];
                if bb == 0 { continue; }

                let mut tmp = bb;
                while tmp != 0 {
                    let sq = Self::lsb_index(tmp);
                    let x = (sq % 9) as usize;
                    let y = (sq / 9) as usize;

                    // stm perspective
                    let stm_base = if c == stm as usize { 0 } else { 630 };
                    let stm_sq_idx = y * 9 + x;
                    let stm_feature = stm_base + pt * 90 + stm_sq_idx;
                    stm_data[stm_feature] = 1.0;

                    tmp &= tmp - 1; // Clear LSB
                }
            }
        }
    }

    /// Count total non-king pieces on the board.
    /// Used for bucket index in NNUE value head.
    pub fn count_non_king_pieces(&self) -> u8 {
        let mut count = 0u8;
        for pt in 0..7 {
            if pt == PieceType::King as usize { continue; }
            count += self.pieces[pt][0].count_ones() as u8;
            count += self.pieces[pt][1].count_ones() as u8;
        }
        count
    }

    /// Find the position of the king of the given color using bitscan.
    /// Returns None if the king has been captured.
    #[inline(always)]
    pub fn king_pos(&self, color: Color) -> Option<Coord> {
        let king_bb = self.pieces[PieceType::King as usize][color as usize];
        if king_bb == 0 {
            return None;
        }
        let sq = Self::lsb_index(king_bb);
        Some(Coord::new((sq % 9) as i8, (sq / 9) as i8))
    }

    /// Find positions of both kings using bitscan.
    /// Returns (red_king_pos, black_king_pos) - either may be None if captured.
    #[inline(always)]
    pub fn find_kings(&self) -> (Option<Coord>, Option<Coord>) {
        (self.king_pos(Color::Red), self.king_pos(Color::Black))
    }
}

impl Default for Bitboards {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, RuleSet, movegen};

    #[test]
    fn test_sq_from_coord_roundtrip() {
        for y in 0..10 {
            for x in 0..9 {
                let sq = sq_from_coord(x, y);
                let coord = coord_from_sq(sq);
                assert_eq!(coord.x, x, "x mismatch for ({x}, {y})");
                assert_eq!(coord.y, y, "y mismatch for ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_bitboards_from_cells_matches_initial_board() {
        let cells = Board::new(RuleSet::Official, 1).cells();
        let bb = Bitboards::from_cells(&cells);

        let red_occ = bb.occupied(Color::Red);
        let black_occ = bb.occupied(Color::Black);
        assert!(red_occ != 0, "Red should have pieces");
        assert!(black_occ != 0, "Black should have pieces");
        assert_eq!(red_occ & black_occ, 0, "Red and Black should not overlap");
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_bitboards_occupied_all_matches_cells() {
        let cells = Board::new(RuleSet::Official, 1).cells();
        let bb = Bitboards::from_cells(&cells);

        let mut cell_count = 0usize;
        for y in 0..10 {
            for x in 0..9 {
                if cells[y][x].is_some() {
                    cell_count += 1;
                }
            }
        }

        let all_occ = bb.occupied_all();
        let bb_count = all_occ.count_ones() as usize;
        assert_eq!(cell_count, bb_count);
    }

    #[test]
    fn test_bitboards_piece_at_matches_cells() {
        let board = Board::new(RuleSet::Official, 1);
        let bb = Bitboards::from_cells(&board.cells());

        for y in 0..10 {
            for x in 0..9 {
                let coord = Coord::new(x, y);
                let cell_piece = board.get(coord);
                let sq = sq_from_coord(x, y);
                let bb_piece = bb.piece_at(sq);
                assert_eq!(cell_piece, bb_piece, "Piece mismatch at ({x}, {y})");
            }
        }
    }

    #[test]
    #[allow(unused_variables)]
    fn test_bitboards_apply_and_undo_move() {
        let mut board = Board::new(RuleSet::Official, 1);
        let moves = movegen::generate_legal_moves(&mut board, Color::Red);

        if let Some(first_move) = moves.first() {
            let src_sq = sq_from_coord(first_move.src.x, first_move.src.y);
            let dst_sq = sq_from_coord(first_move.tar.x, first_move.tar.y);
            let piece = board.get(first_move.src).unwrap();
            let captured = first_move.captured;

            let mut bb = Bitboards::from_cells(&board.cells());
            let before_occ = bb.occupied_all();

            bb.apply_move(src_sq, dst_sq, captured, piece);
            let after_apply = bb.occupied_all();
            assert_ne!(before_occ, after_apply, "Bitboards should change after apply_move");

            bb.undo_move(src_sq, dst_sq, captured, piece);
            let after_undo = bb.occupied_all();
            assert_eq!(before_occ, after_undo, "Bitboards should restore after undo_move");
        }
    }

    #[test]
    fn test_bitboards_sync_after_make_and_undo() {
        let mut board = Board::new(RuleSet::Official, 1);
        let moves = movegen::generate_legal_moves(&mut board, Color::Red);

        if let Some(first_move) = moves.first() {
            let action = *first_move;

            // Before move: bitboards should match cells
            let bb_before = Bitboards::from_cells(&board.cells());
            for y in 0..10 {
                for x in 0..9 {
                    let coord = Coord::new(x, y);
                    let sq = (y * 9 + x) as u8;
                    assert_eq!(board.get(coord), bb_before.piece_at(sq), "Pre-move mismatch at ({x}, {y})");
                }
            }

            board.make_move(action);

            // After move: cells changed, bitboards should be in sync
            for y in 0..10 {
                for x in 0..9 {
                    let coord = Coord::new(x, y);
                    let sq = (y * 9 + x) as u8;
                    assert_eq!(board.get(coord), board.bitboards.piece_at(sq), "Post-move mismatch at ({x}, {y})");
                }
            }

            // Undo and verify
            board.undo_move(action);
            let bb_after_undo = Bitboards::from_cells(&board.cells());
            for y in 0..10 {
                for x in 0..9 {
                    let coord = Coord::new(x, y);
                    let sq = (y * 9 + x) as u8;
                    assert_eq!(board.get(coord), bb_after_undo.piece_at(sq), "Post-undo mismatch at ({x}, {y})");
                }
            }
        }
    }

    #[cfg(feature = "nnue")]
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_bitboards_nnue_equivalence() {
        use crate::nnue_input::NNInputPlanes;
        use crate::movegen::generate_legal_moves;

        // Test on initial position
        let board = Board::new(RuleSet::Official, 1);
        let bb = &board.bitboards;

        let mut stm_data = [0.0f32; 1260];
        bb.fill_nnue_features(board.current_side, &mut stm_data);

        let (expected_stm, _expected_ntm) = NNInputPlanes::from_board(&board);

        for i in 0..1260 {
            assert_eq!(stm_data[i], expected_stm.data[i], "NNUE feature mismatch at index {}", i);
        }

        // Test after a few random moves
        let mut board2 = Board::new(RuleSet::Official, 1);
        for _ in 0..5 {
            let side = board2.current_side;
            let moves = generate_legal_moves(&mut board2, side);
            if moves.is_empty() { break; }
            let idx = board2.zobrist_key as usize % moves.len();
            board2.make_move(moves[idx]);
        }

        let bb2 = &board2.bitboards;
        let mut stm_data2 = [0.0f32; 1260];
        bb2.fill_nnue_features(board2.current_side, &mut stm_data2);

        let (expected_stm2, _expected_ntm2) = NNInputPlanes::from_board(&board2);
        for i in 0..1260 {
            assert_eq!(stm_data2[i], expected_stm2.data[i], "NNUE feature mismatch after moves at index {}", i);
        }
    }

    #[test]
    fn test_count_non_king_pieces() {
        let board = Board::new(RuleSet::Official, 1);
        let bb = Bitboards::from_cells(&board.cells());

        let count = bb.count_non_king_pieces();
        // Initial position: 32 pieces total, 2 kings -> 30 non-kings
        assert_eq!(count, 30, "Initial position should have 30 non-king pieces");
    }

    // =============================================================================
    // BUG 1: chariot_attacks — lsb_index gives wrong blocker for SOUTH/WEST directions
    //
    // Root cause: init_chariot_rays sets bits in direction-dependent order:
    //   North (+y): bits added near→far (54,63,72...), so lsb_index = nearest ✓
    //   South (-y): bits added far→near (36,27,18,9,0...), so lsb_index = FURTHEST ✗
    //   East  (+x): bits added near→far (46,47,48...), so lsb_index = nearest ✓
    //   West  (-x): bits added far→near (44,43,42...), so lsb_index = FURTHEST ✗
    // =============================================================================

    /// Chariot at (4,5)=45 attacks SOUTH. Friendly at 36 (nearest), enemy at 27 (further).
    /// Bug: lsb_index({36,27}) = 27 (the FURTHEST piece, not nearest!)
    /// Expected: friendly at 36 blocks → no attack squares south
    #[test]
    fn test_chariot_attacks_south_blocked_by_nearest() {
        let mut bb = Bitboards::new();
        // Red chariot at (4,5) = 45
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        // Red friendly pawn at (4,4) = 36 (one square south — nearest blocker)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 36;
        // Black enemy pawn at (4,3) = 27 (two squares south — beyond friendly)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 27;

        let attacks = bb.chariot_attacks(45, Color::Red);

        // Friendly at 36 should block completely — nothing south of it reachable
        assert_eq!(attacks & (1_u128 << 36), 0,
            "Friendly at 36 should block — no attack south");
        assert_eq!(attacks & (1_u128 << 27), 0,
            "Enemy at 27 should NOT be attacked (blocked by friendly at 36)");
    }

    /// Chariot at (4,5)=45 attacks WEST. Friendly at 44 (nearest), enemy at 43 (further).
    /// Bug: lsb_index({44,43}) = 43 (the FURTHEST piece, not nearest!)
    /// Expected: friendly at 44 blocks → no attack squares west
    #[test]
    fn test_chariot_attacks_west_blocked_by_nearest() {
        let mut bb = Bitboards::new();
        // Red chariot at (4,5) = 45
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        // Red friendly at (3,5) = 44 (one square west — nearest blocker)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 44;
        // Black enemy at (2,5) = 43 (two squares west)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 43;

        let attacks = bb.chariot_attacks(45, Color::Red);

        // Friendly at 44 should block — nothing west reachable
        assert_eq!(attacks & (1_u128 << 44), 0,
            "Friendly at 44 should block — no attack west");
        assert_eq!(attacks & (1_u128 << 43), 0,
            "Enemy at 43 should NOT be attacked (blocked by friendly at 44)");
    }

    /// Chariot at (4,5)=45 attacks NORTH — this direction works correctly (near-to-far).
    #[test]
    fn test_chariot_attacks_north_blocked_correctly() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        // Red friendly at (4,6) = 54 (one square north — nearest blocker)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 54;
        // Black enemy at (4,7) = 63 (two squares north)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 63;

        let attacks = bb.chariot_attacks(45, Color::Red);

        // North: lsb_index correctly gives 54 (nearest), so it blocks properly
        assert_eq!(attacks & (1_u128 << 54), 0, "Friendly at 54 should block north");
        assert_eq!(attacks & (1_u128 << 63), 0, "Enemy at 63 blocked by friendly at 54");
    }

    /// Chariot at (4,5)=45 attacks EAST — this direction works correctly (near-to-far).
    #[test]
    fn test_chariot_attacks_east_blocked_correctly() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        // Red friendly at (5,5) = 46 (one square east — nearest blocker)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 46;
        // Black enemy at (6,5) = 47 (two squares east)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 47;

        let attacks = bb.chariot_attacks(45, Color::Red);

        assert_eq!(attacks & (1_u128 << 46), 0, "Friendly at 46 should block east");
        assert_eq!(attacks & (1_u128 << 47), 0, "Enemy at 47 blocked by friendly at 46");
    }

    /// Chariot at (4,5)=45 with no blockers — should reach all 17 squares.
    #[test]
    fn test_unblocked_chariot_has_17_destinations() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;

        let attacks = bb.chariot_attacks(45, Color::Red);
        let count = attacks.count_ones();

        // From (4,5): North has 4 squares (y=6,7,8,9),
        // South has 5 squares (y=4,3,2,1,0),
        // East has 4 squares (x=5,6,7,8),
        // West has 4 squares (x=3,2,1,0) = 4+5+4+4 = 17
        assert_eq!(count, 17, "Unblocked chariot at (4,5) should reach 17 squares, got {}", count);
    }

    // =============================================================================
    // BUG 2: cannon_attacks — same lsb_index direction bug
    // =============================================================================

    /// Cannon at (4,5)=45 attacks SOUTH with screen at 36 and enemy at 27.
    /// Bug: lsb_index({36,27}) = 27 incorrectly identifies the screen.
    #[test]
    fn test_cannon_attacks_south_one_screen_capture() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        // Red screen at (4,4) = 36 (one square south)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 36;
        // Black enemy at (4,3) = 27 (two squares south)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 27;

        let attacks = bb.cannon_attacks(45, Color::Red);

        // Cannon with one screen should capture enemy at 27
        assert_ne!(attacks & (1_u128 << 27), 0,
            "Cannon should capture enemy at 27 with one screen at 36");
    }

    // =============================================================================
    // BUG 3: attackers() — lsb_index direction bug + missing Advisor/Elephant
    // =============================================================================

    /// Black chariot at 27 (x=4,y=3) attacks north toward red at 45 — blocked by red at 36.
    #[test]
    fn test_attackers_chariot_north_blocked() {
        let mut bb = Bitboards::new();
        // Black chariot at (4,3) = 27
        bb.pieces[PieceType::Chariot as usize][Color::Black as usize] = 1_u128 << 27;
        // Red chariot at (4,5) = 45 (target)
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        // Red pawn at (4,4) = 36 (blocks north attack from 27 to 45)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 36;

        let attackers = bb.attackers(45, Color::Black);

        // Chariot at 27 looking north: ray = {36,45,54,...}, nearest = 36 (red) → blocks
        assert_eq!(attackers & (1_u128 << 27), 0,
            "Black chariot at 27 should NOT attack 45 (blocked by red at 36)");
    }

    /// Black chariot at 45 attacks south toward red at 27 — blocked by red at 36.
    #[test]
    fn test_attackers_chariot_south_blocked() {
        let mut bb = Bitboards::new();
        // Black chariot at (4,5) = 45
        bb.pieces[PieceType::Chariot as usize][Color::Black as usize] = 1_u128 << 45;
        // Red at (4,4) = 36 (blocks southward attack from 45 to 27)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 36;
        // Red at (4,3) = 27 (target)
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] |= 1_u128 << 27;

        let attackers = bb.attackers(27, Color::Black);

        // Chariot at 45 looking south: ray = {36,27,18,...}, nearest = 36 (red) → blocks
        assert_eq!(attackers & (1_u128 << 45), 0,
            "Black chariot at 45 should NOT attack 27 (blocked by red at 36)");
    }

    /// Advisor attack should be detected by attackers()
    #[test]
    fn test_attackers_advisor() {
        let mut bb = Bitboards::new();
        // Black advisor at (5,2) = 23 (in black palace: y<=2)
        bb.pieces[PieceType::Advisor as usize][Color::Black as usize] = 1_u128 << 23;
        // Red pawn at (4,1) = 13 (diagonal from advisor — target)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 13;

        let attackers = bb.attackers(13, Color::Black);

        // Black advisor at 23 should attack 13 (diagonal within palace)
        // (5,2) → (4,1) is one diagonal step
        // Bug: Advisor is NOT checked in attackers()
        assert_ne!(attackers & (1_u128 << 23), 0,
            "Black advisor at 23 should attack red pawn at 13");
    }

    /// Elephant attack should be detected by attackers()
    #[test]
    fn test_attackers_elephant() {
        let mut bb = Bitboards::new();
        // Black elephant at (2,0) = 18 (can move to (0,2)=2 via (-2,2), eye at (1,1)=10)
        bb.pieces[PieceType::Elephant as usize][Color::Black as usize] = 1_u128 << 18;
        // Red pawn at (0,2) = 2 (target square — diagonal from elephant)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 2;

        let attackers = bb.attackers(2, Color::Black);

        // Black elephant at 18 should attack 2 (diagonal within black's side)
        // Bug: Elephant is NOT checked in attackers()
        assert_ne!(attackers & (1_u128 << 18), 0,
            "Black elephant at 18 should attack red pawn at 2");
    }

    // =============================================================================
    // Verify chariot rays storage
    // =============================================================================

    #[test]
    fn test_chariot_rays_south_count() {
        let rays = get_chariot_rays();
        // For sq=45 (x=4, y=5), direction 1 (South): y=4,3,2,1,0 → 36,27,18,9,0 (5 squares)
        let south = rays[45][1];
        let count = south.count_ones();
        assert_eq!(count, 5, "South ray from (4,5) should have 5 squares, got {}", count);
    }

    #[test]
    fn test_chariot_rays_north_count() {
        let rays = get_chariot_rays();
        // For sq=45 (x=4, y=5), direction 0 (North): y=6,7,8,9 → 54,63,72,81 (4 squares)
        let north = rays[45][0];
        let count = north.count_ones();
        assert_eq!(count, 4, "North ray from (4,5) should have 4 squares, got {}", count);
    }

    #[test]
    fn test_chariot_rays_south_includes_correct_squares() {
        let rays = get_chariot_rays();
        let south = rays[45][1];
        // Should include 36,27,18,9,0 but NOT 45
        assert_ne!(south & (1_u128 << 36), 0);
        assert_ne!(south & (1_u128 << 27), 0);
        assert_ne!(south & (1_u128 << 18), 0);
        assert_ne!(south & (1_u128 << 9), 0);
        assert_ne!(south & (1_u128 << 0), 0);
        assert_eq!(south & (1_u128 << 45), 0, "Source square 45 should not be in ray");
    }

    // =============================================================================
    // generate_moves and generate_pseudo_moves
    // =============================================================================

    #[test]
    fn test_generate_pseudo_moves_unblocked_chariot_count() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;

        let moves = bb.generate_pseudo_moves(Color::Red);

        // From (4,5) on empty board: 4 north + 5 south + 4 east + 4 west = 17 moves
        assert_eq!(moves.len(), 17, "Unblocked chariot should generate 17 pseudo-moves, got {}", moves.len());
    }

    #[test]
    fn test_generate_moves_chariot_includes_captures() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 63;

        let moves = bb.generate_moves(45, Color::Red);
        let dsts: u128 = moves.iter().fold(0u128, |acc, &d| acc | (1_u128 << d));

        // Should include 63 (enemy capture)
        assert_ne!(dsts & (1_u128 << 63), 0, "Chariot should have enemy at 63 as destination");
    }

    // =============================================================================
    // PAWN ATTACK TESTS (coordinate: (0,0) top-left, y increases down)
    // =============================================================================

    /// Red pawn at (4,6)=58 advances toward y=0 (up/forward for Red)
    /// Red dir=-1, so forward is y-1
    #[test]
    fn test_red_pawn_forward_attack() {
        let mut bb = Bitboards::new();
        // Red pawn at (4,6) = 4 + 6*9 = 58
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 58;

        let attacks = bb.pawn_attacks(58, Color::Red);
        // Forward: y-1=5, sq = 4 + 5*9 = 49
        assert_ne!(attacks & (1_u128 << 49), 0,
            "Red pawn at (4,6) should attack forward to (4,5)=49");
    }

    /// Black pawn at (4,3)=31 advances toward y=9 (down/forward for Black)
    /// Black dir=+1, so forward is y+1
    #[test]
    fn test_black_pawn_forward_attack() {
        let mut bb = Bitboards::new();
        // Black pawn at (4,3) = 4 + 3*9 = 31 (NOT 30!)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 31;

        let attacks = bb.pawn_attacks(31, Color::Black);
        // Forward: y+1=4, sq = 4 + 4*9 = 40
        assert_ne!(attacks & (1_u128 << 40), 0,
            "Black pawn at (4,3) should attack forward to (4,4)=40");
    }

    /// Red pawn at y=5 (before river at y=4) should NOT have side attacks yet
    /// Red crosses when y <= 4
    #[test]
    fn test_red_pawn_before_river_no_side() {
        let mut bb = Bitboards::new();
        // Red pawn at (4,5) = 4 + 5*9 = 49
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 49;

        let attacks = bb.pawn_attacks(49, Color::Red);
        // y=5 > 4, hasn't crossed river yet - only forward (to y=4)
        assert_eq!(attacks.count_ones(), 1,
            "Red pawn at y=5 should only have forward attack, no side");
    }

    /// Red pawn at y=4 (AT river, has crossed) should have side attacks
    #[test]
    fn test_red_pawn_at_river_has_side() {
        let mut bb = Bitboards::new();
        // Red pawn at (4,4) = 4 + 4*9 = 40
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 40;

        let attacks = bb.pawn_attacks(40, Color::Red);
        // y=4 <= 4, has crossed - forward to y=3 + side to x=3,5
        // Forward: (4,3)=31, Side: (3,4)=37, (5,4)=41
        assert!(attacks.count_ones() >= 2,
            "Red pawn at y=4 should have forward + side attacks");
    }

    /// Black pawn at y=4 (before river at y=5) should NOT have side attacks yet
    /// Black crosses when y >= 5
    #[test]
    fn test_black_pawn_before_river_no_side() {
        let mut bb = Bitboards::new();
        // Black pawn at (4,4) = 4 + 4*9 = 40
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 40;

        let attacks = bb.pawn_attacks(40, Color::Black);
        // y=4 < 5, hasn't crossed yet - only forward (to y=5)
        assert_eq!(attacks.count_ones(), 1,
            "Black pawn at y=4 should only have forward attack");
    }

    /// Black pawn at y=5 (AT river, has crossed) should have side attacks
    #[test]
    fn test_black_pawn_at_river_has_side() {
        let mut bb = Bitboards::new();
        // Black pawn at (4,5) = 4 + 5*9 = 49
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 49;

        let attacks = bb.pawn_attacks(49, Color::Black);
        // y=5 >= 5, has crossed - forward to y=6 + side
        assert!(attacks.count_ones() >= 2,
            "Black pawn at y=5 should have forward + side attacks");
    }

    // =============================================================================
    // ELEPHANT ATTACK TESTS
    // Red elephant at y>=5 stays on own side (cannot cross to y<=4)
    // Black elephant at y<=4 stays on own side (cannot cross to y>=5)
    // =============================================================================

    /// Red elephant at (2,9)=83 can attack (4,7)=67 (y=7>4, on own side)
    #[test]
    fn test_red_elephant_attacks_own_side() {
        let mut bb = Bitboards::new();
        // Red elephant at (2,9) = 2 + 9*9 = 83
        bb.pieces[PieceType::Elephant as usize][Color::Red as usize] = 1_u128 << 83;

        let attacks = bb.elephant_attacks(83, Color::Red);
        // (4,7)=4+7*9=67 has y=7 > 4, OK for Red
        assert_ne!(attacks & (1_u128 << 67), 0,
            "Red elephant at (2,9) should attack (4,7)");
    }

    /// Red elephant at (2,9)=83 cannot attack (4,3)=31 (y=3<=4, would cross river)
    #[test]
    fn test_red_elephant_cannot_cross_river() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Elephant as usize][Color::Red as usize] = 1_u128 << 83;

        let attacks = bb.elephant_attacks(83, Color::Red);
        // (4,3)=31 has y=3 <= 4, crosses river for Red
        assert_eq!(attacks & (1_u128 << 31), 0,
            "Red elephant should NOT attack (4,3) - crosses river");
    }

    /// Black elephant at (2,0)=2 can attack (4,2)=22 (y=2<5, on own side)
    #[test]
    fn test_black_elephant_attacks_own_side() {
        let mut bb = Bitboards::new();
        // Black elephant at (2,0) = 2 + 0*9 = 2
        bb.pieces[PieceType::Elephant as usize][Color::Black as usize] = 1_u128 << 2;

        let attacks = bb.elephant_attacks(2, Color::Black);
        // (4,2)=22 has y=2 < 5, OK for Black
        assert_ne!(attacks & (1_u128 << 22), 0,
            "Black elephant at (2,0) should attack (4,2)");
    }

    /// Black elephant at (2,0)=2 cannot attack (4,6)=58 (y=6>=5, would cross)
    #[test]
    fn test_black_elephant_cannot_cross_river() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Elephant as usize][Color::Black as usize] = 1_u128 << 2;

        let attacks = bb.elephant_attacks(2, Color::Black);
        // (4,6)=58 has y=6 >= 5, crosses river for Black
        assert_eq!(attacks & (1_u128 << 58), 0,
            "Black elephant should NOT attack (4,6) - crosses river");
    }

    /// Elephant eye must be empty - blocking one direction still leaves others
    #[test]
    fn test_elephant_eye_blocked() {
        let mut bb = Bitboards::new();
        // Red elephant at (2,9)=83
        bb.pieces[PieceType::Elephant as usize][Color::Red as usize] = 1_u128 << 83;
        // Block BOTH valid eyes to fully block the elephant:
        // Eye (3,8)=75 for delta (+2,-2) → target (4,7)
        // Eye (1,8)=73 for delta (-2,-2) → target (0,7)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = (1_u128 << 75) | (1_u128 << 73);

        let attacks = bb.elephant_attacks(83, Color::Red);
        assert_eq!(attacks, 0,
            "Elephant with both valid eyes blocked should have no attacks");
    }

    // =============================================================================
    // ADVISOR ATTACK TESTS
    // Red palace: y>=7, Black palace: y<=2
    // =============================================================================

    /// Red advisor at (3,7)=66 can attack (4,8)=76 (y=8>=7, in palace)
    #[test]
    fn test_red_advisor_in_palace() {
        let mut bb = Bitboards::new();
        // Red advisor at (3,7) = 3 + 7*9 = 66
        bb.pieces[PieceType::Advisor as usize][Color::Red as usize] = 1_u128 << 66;

        let attacks = bb.advisor_attacks(66, Color::Red);
        // (4,8)=76 has y=8 >= 7, in palace
        assert_ne!(attacks & (1_u128 << 76), 0,
            "Red advisor at (3,7) should attack (4,8)");
    }

    /// Red advisor at (3,7)=66 cannot attack (4,6)=58 (y=6<7, out of palace)
    #[test]
    fn test_red_advisor_out_of_palace() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Advisor as usize][Color::Red as usize] = 1_u128 << 66;

        let attacks = bb.advisor_attacks(66, Color::Red);
        // (4,6)=58 has y=6 < 7, out of palace
        assert_eq!(attacks & (1_u128 << 58), 0,
            "Red advisor should NOT attack (4,6) - out of palace");
    }

    /// Black advisor at (4,2)=22 can attack (5,3)=32 (y=3<=2? No! y=3>2)
    /// Wait, (4,2) is NOT in black's palace (y=2, x=4). Black palace is y<=2, x=3-5
    /// So (4,2) is at the edge. (5,3) has y=3 > 2, NOT in black's palace.
    /// Let me use (5,2)=23 which IS in black's palace.
    #[test]
    fn test_black_advisor_in_palace() {
        let mut bb = Bitboards::new();
        // Black advisor at (5,2) = 5 + 2*9 = 23
        bb.pieces[PieceType::Advisor as usize][Color::Black as usize] = 1_u128 << 23;

        let attacks = bb.advisor_attacks(23, Color::Black);
        // (4,3)=31 has y=3 > 2, NOT in palace. (4,1)=13 is in palace.
        assert_ne!(attacks & (1_u128 << 13), 0,
            "Black advisor at (5,2) should attack (4,1)");
    }

    // =============================================================================
    // KING ATTACK TESTS
    // =============================================================================

    /// Red king at (4,7)=67 can attack (4,8)=76 (y=8>=7, in palace)
    #[test]
    fn test_red_king_in_palace() {
        let mut bb = Bitboards::new();
        // Red king at (4,7) = 4 + 7*9 = 67
        bb.pieces[PieceType::King as usize][Color::Red as usize] = 1_u128 << 67;

        let attacks = bb.king_attacks(67, Color::Red);
        assert_ne!(attacks & (1_u128 << 76), 0,
            "Red king at (4,7) should attack (4,8)");
    }

    /// Red king at (4,7)=67 cannot attack (4,6)=58 (y=6<7, out of palace)
    #[test]
    fn test_red_king_out_of_palace() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::King as usize][Color::Red as usize] = 1_u128 << 67;

        let attacks = bb.king_attacks(67, Color::Red);
        assert_eq!(attacks & (1_u128 << 58), 0,
            "Red king should NOT attack (4,6) - out of palace");
    }

    /// Black king at (4,2)=22 can attack (4,1)=13 (y=1<=2, in palace)
    #[test]
    fn test_black_king_in_palace() {
        let mut bb = Bitboards::new();
        // Black king at (4,2) = 4 + 2*9 = 22
        bb.pieces[PieceType::King as usize][Color::Black as usize] = 1_u128 << 22;

        let attacks = bb.king_attacks(22, Color::Black);
        assert_ne!(attacks & (1_u128 << 13), 0,
            "Black king at (4,2) should attack (4,1)");
    }

    // =============================================================================
    // CANNON ATTACK TESTS
    // =============================================================================

    /// Cannon with no screen cannot capture
    #[test]
    fn test_cannon_no_screen_no_capture() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 63;

        let attacks = bb.cannon_attacks(45, Color::Red);
        // No screen in north direction, so no capture
        assert_eq!(attacks & (1_u128 << 63), 0,
            "Cannon with no screen should not capture");
    }

    /// Cannon with exactly one screen CAN capture
    #[test]
    fn test_cannon_one_screen_capture() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 54; // screen
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 63; // target

        let attacks = bb.cannon_attacks(45, Color::Red);
        assert_ne!(attacks & (1_u128 << 63), 0,
            "Cannon with one screen should capture");
    }

    /// TEST FOR BUG: cannon_attackers should find cannon NOT at screen position
    /// Cannon at 45 (Red) attacks target 63 through screen at 54 (Red pawn)
    #[test]
    fn test_attackers_cannon_through_screen() {
        let mut bb = Bitboards::new();
        // Cannon at (5,5) = 45 (Red)
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        // Screen at (6,5) = 54 (Red pawn - screen between cannon and target)
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 54;
        // Target at (7,5) = 63 (Black pawn - what cannon should capture)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 63;

        let attackers = bb.attackers(63, Color::Red);

        // Red cannon at 45 should be able to attack 63 through screen at 54
        assert_ne!(attackers & (1_u128 << 45), 0,
            "Red cannon at 45 should attack Black pawn at 63 through screen at 54");
    }

    /// Cannon attacks: screen can be ANY color (even opponent's piece)
    #[test]
    fn test_attackers_cannon_with_opponent_screen() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        // Screen at (6,5) = 54 (BLACK pawn - opponent's screen)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 54;
        // Target at (7,5) = 63 (Black pawn)
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] |= 1_u128 << 63;

        let attackers = bb.attackers(63, Color::Red);
        assert_ne!(attackers & (1_u128 << 45), 0,
            "Red cannon at 45 should attack through Black screen at 54");
    }

    /// Black pawn at (4,3) should be able to capture Red pawn at (4,4)
    #[test]
    fn test_black_pawn_at_4_3_capture_red_at_4_4() {
        let mut bb = Bitboards::new();
        // Red pawn at (4, 4) - x=4, y=4 -> sq = 4*9 + 4 = 40
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 40;
        // Black pawn at (4, 3) - x=4, y=3 -> sq = 3*9 + 4 = 31
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 31;

        // Test pawn_attacks from Black's perspective
        let attacks = bb.pawn_attacks(31, Color::Black);
        // Black pawn at y=3, dir=+1, forward attacks to y=4 -> sq=31+9=40
        assert_ne!(attacks & (1_u128 << 40), 0,
            "Black pawn at (4,3) should attack forward to (4,4)");

        // Test attackers function
        let attackers = bb.attackers(40, Color::Black);
        assert_ne!(attackers & (1_u128 << 31), 0,
            "Black pawn at (4,3) should be attacker of Red pawn at (4,4)");
    }

    /// Cannon with two screens cannot capture
    #[test]
    fn test_cannon_two_screens_no_capture() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Cannon as usize][Color::Red as usize] = 1_u128 << 45;
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] = 1_u128 << 54; // screen 1
        bb.pieces[PieceType::Pawn as usize][Color::Red as usize] |= 1_u128 << 63; // screen 2

        let attacks = bb.cannon_attacks(45, Color::Red);
        // With 2 screens, no capture in that direction
        assert_eq!(attacks & (1_u128 << 72), 0,
            "Cannon with two screens should not capture");
    }

    // =============================================================================
    // APPLY_MOVE / UNDO_MOVE TESTS
    // =============================================================================

    #[test]
    fn test_apply_undo_preserves_position() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;

        let before = bb.pieces;
        bb.apply_move(45, 54, None, Piece { color: Color::Red, piece_type: PieceType::Chariot });

        assert_eq!(bb.pieces[PieceType::Chariot as usize][Color::Red as usize] & (1_u128 << 45), 0,
            "Chariot should be removed from 45");
        assert_ne!(bb.pieces[PieceType::Chariot as usize][Color::Red as usize] & (1_u128 << 54), 0,
            "Chariot should be at 54");

        bb.undo_move(45, 54, None, Piece { color: Color::Red, piece_type: PieceType::Chariot });
        assert_eq!(bb.pieces, before,
            "After undo, bitboards should be exactly restored");
    }

    #[test]
    fn test_apply_undo_capture_restores() {
        let mut bb = Bitboards::new();
        bb.pieces[PieceType::Chariot as usize][Color::Red as usize] = 1_u128 << 45;
        bb.pieces[PieceType::Pawn as usize][Color::Black as usize] = 1_u128 << 54;

        let before = bb.pieces;
        bb.apply_move(45, 54, Some(Piece { color: Color::Black, piece_type: PieceType::Pawn }),
                    Piece { color: Color::Red, piece_type: PieceType::Chariot });

        assert_eq!(bb.pieces[PieceType::Pawn as usize][Color::Black as usize], 0,
            "Black pawn should be captured");

        bb.undo_move(45, 54, Some(Piece { color: Color::Black, piece_type: PieceType::Pawn }),
                    Piece { color: Color::Red, piece_type: PieceType::Chariot });
        assert_eq!(bb.pieces, before,
            "After undo, captured piece should be restored");
    }
}