//! Bitboard representation for Xiangqi (Chinese Chess).
//!
//! Uses u128 per piece type per color (14 total bitboards for 7 piece types × 2 colors).
//! Board is 9×10 = 90 squares, bits 90-127 unused (always 0).
//!
//! Bit index formula: `sq = y * 9 + x` where x=0-8, y=0-9

use crate::{Color, Piece, PieceType, Coord};
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Number of squares on the Xiangqi board
pub const BOARD_SQ_COUNT: usize = 90;

/// Convert board Coord to bitboard square index (0-89)
#[inline(always)]
pub fn sq_from_coord(x: i8, y: i8) -> u8 {
    (y * 9 + x) as u8
}

/// Convert bitboard square index (0-89) to Coord
#[inline(always)]
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
static CANNON_SCREENS_STORAGE: OnceLock<[[u128; 4]; BOARD_SQ_COUNT]> = OnceLock::new();
static HORSE_ATTACKS_STORAGE: OnceLock<[[u128; 8]; BOARD_SQ_COUNT]> = OnceLock::new();
static ADVISOR_ATTACKS_STORAGE: OnceLock<[[u128; 4]; BOARD_SQ_COUNT]> = OnceLock::new();
static ELEPHANT_ATTACKS_STORAGE: OnceLock<[[u128; 4]; BOARD_SQ_COUNT]> = OnceLock::new();
static KING_ATTACKS_STORAGE: OnceLock<[u128; BOARD_SQ_COUNT]> = OnceLock::new();

/// Initialize CHARIOT_RAYS and CANNON_SCREENS tables.
/// Each entry is a u128 mask of squares along a ray from sq in one direction.
fn init_chariot_rays() -> ([[u128; 4]; BOARD_SQ_COUNT], [[u128; 4]; BOARD_SQ_COUNT]) {
    let mut chariot_rays = [[0u128; 4]; BOARD_SQ_COUNT];
    let mut cannon_screens = [[0u128; 4]; BOARD_SQ_COUNT];

    for sq in 0..BOARD_SQ_COUNT {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;

        // Direction 0: North (+y, toward y=9)
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

pub fn init_non_slide_attacks() -> (
    [[u128; 8]; BOARD_SQ_COUNT],
    [[u128; 4]; BOARD_SQ_COUNT],
    [[u128; 4]; BOARD_SQ_COUNT],
    [u128; BOARD_SQ_COUNT],
) {
    let mut horse_attacks = [[0u128; 8]; BOARD_SQ_COUNT];
    let mut advisor_attacks = [[0u128; 4]; BOARD_SQ_COUNT];
    let mut elephant_attacks = [[0u128; 4]; BOARD_SQ_COUNT];
    let mut king_attacks = [0u128; BOARD_SQ_COUNT];

    // Horse deltas (8 directions, knee position checked separately by caller)
    let horse_deltas: [(i8, i8); 8] = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)];
    // Elephant deltas and eye positions (4 diagonal directions)
    let elephant_deltas: [(i8, i8); 4] = [(2, 2), (2, -2), (-2, 2), (-2, -2)];
    let elephant_blocks: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    // Advisor deltas (4 diagonal directions, palace-bound)
    let advisor_deltas: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    // King orthogonal offsets (palace-bound)
    let king_offsets: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

    for sq in 0..BOARD_SQ_COUNT {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;

        // Horse: 8 L-shape destinations (knee position checked separately by caller)
        for (i, &(dx, dy)) in horse_deltas.iter().enumerate() {
            let tx = x + dx;
            let ty = y + dy;
            if (0..9).contains(&tx) && (0..10).contains(&ty) {
                let tsq = (ty * 9 + tx) as u8;
                horse_attacks[sq][i] = 1_u128 << tsq;
            }
        }

        // Advisor: 4 diagonal destinations (palace-bound checked by caller)
        for (i, &(dx, dy)) in advisor_deltas.iter().enumerate() {
            let tx = x + dx;
            let ty = y + dy;
            if (0..9).contains(&tx) && (0..10).contains(&ty) {
                let tsq = (ty * 9 + tx) as u8;
                advisor_attacks[sq][i] = 1_u128 << tsq;
            }
        }

        // Elephant: 4 diagonal destinations (river-bound + eye-check done by caller)
        for (i, ((dx, dy), (bx, by))) in elephant_deltas.iter().zip(elephant_blocks.iter()).enumerate() {
            let tx = x + dx;
            let ty = y + dy;
            let ex = x + bx;
            let ey = y + by;
            if (0..9).contains(&tx) && (0..10).contains(&ty)
                && (0..9).contains(&ex) && (0..10).contains(&ey)
            {
                let tsq = (ty * 9 + tx) as u8;
                elephant_attacks[sq][i] = 1_u128 << tsq;
            }
        }

        // King: 4 orthogonal destinations (palace-bound checked by caller)
        for (dx, dy) in king_offsets {
            let tx = x + dx;
            let ty = y + dy;
            if (0..9).contains(&tx) && (0..10).contains(&ty) {
                let tsq = (ty * 9 + tx) as u8;
                king_attacks[sq] |= 1_u128 << tsq;
            }
        }
    }

    (horse_attacks, advisor_attacks, elephant_attacks, king_attacks)
}

pub fn get_horse_attacks() -> &'static [[u128; 8]; BOARD_SQ_COUNT] {
    HORSE_ATTACKS_STORAGE.get_or_init(|| init_non_slide_attacks().0)
}

pub fn get_advisor_attacks() -> &'static [[u128; 4]; BOARD_SQ_COUNT] {
    ADVISOR_ATTACKS_STORAGE.get_or_init(|| init_non_slide_attacks().1)
}

pub fn get_elephant_attacks() -> &'static [[u128; 4]; BOARD_SQ_COUNT] {
    ELEPHANT_ATTACKS_STORAGE.get_or_init(|| init_non_slide_attacks().2)
}

pub fn get_king_attacks() -> &'static [u128; BOARD_SQ_COUNT] {
    KING_ATTACKS_STORAGE.get_or_init(|| init_non_slide_attacks().3)
}

/// Get CHARIOT_RAYS table (lazily initialized)
pub fn get_chariot_rays() -> &'static [[u128; 4]; BOARD_SQ_COUNT] {
    CHARIOT_RAYS_STORAGE.get_or_init(|| init_chariot_rays().0)
}

/// Get CANNON_SCREENS table (lazily initialized)
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
                    return Some(Piece {
                        color: if c == 0 { Color::Red } else { Color::Black },
                        piece_type: unsafe { std::mem::transmute(pt as u8) },
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
        for y in 0..10 {
            for x in 0..9 {
                cells[y][x] = self.piece_at((y * 9 + x) as u8);
            }
        }
        cells
    }

    /// Flip the bitboards vertically (swap rows 0↔9, 1↔8, 2↔7, 3↔6, 4↔5).
    /// This is used for display purposes when showing the board from Black's perspective.
    pub fn flip_vertically(&mut self) {
        let mut new_pieces = [[0u128; 2]; 7];

        for pt in 0..7 {
            for c in 0..2 {
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
                new_pieces[pt][c] = new_bb;
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
    pub fn chariot_attacks(&self, sq: u8) -> u128 {
        let occ = self.occupied_all();
        let rays = get_chariot_rays();
        let mut attacks = 0u128;

        for dir in 0..4 {
            let ray = rays[sq as usize][dir];
            let blockers = ray & occ;
            if blockers == 0 {
                attacks |= ray;
            } else {
                let nearest = Self::lsb_index(blockers);
                let ray_to_nearest = ray & !(rays[nearest as usize][dir]);
                attacks |= ray_to_nearest;
            }
        }
        attacks
    }

    /// Cannon attacks from sq — slides until first screen, then captures through it.
    pub fn cannon_attacks(&self, sq: u8) -> u128 {
        let occ = self.occupied_all();
        let rays = get_chariot_rays();
        let mut attacks = 0u128;

        for dir in 0..4 {
            let ray = rays[sq as usize][dir];
            let blockers = ray & occ;
            if blockers == 0 {
                // No screen, no captures
            } else {
                let nearest = Self::lsb_index(blockers);
                let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
                if second_blockers != 0 {
                    let second = Self::lsb_index(second_blockers);
                    let capture_ray = ray & !(rays[second as usize][dir]);
                    attacks |= capture_ray;
                }
            }
        }
        attacks
    }

    /// Horse attacks from sq — 8 L-shape destinations (knee square unchecked by this function).
    pub fn horse_attacks(&self, sq: u8) -> u128 {
        get_horse_attacks()[sq as usize].iter().fold(0u128, |acc, &m| acc | m)
    }

    /// Advisor attacks from sq — 4 diagonal destinations (palace-bound checked by caller).
    pub fn advisor_attacks(&self, sq: u8) -> u128 {
        get_advisor_attacks()[sq as usize].iter().fold(0u128, |acc, &m| acc | m)
    }

    /// Elephant attacks from sq — 4 diagonal destinations (river-bound checked by caller).
    pub fn elephant_attacks(&self, sq: u8) -> u128 {
        get_elephant_attacks()[sq as usize].iter().fold(0u128, |acc, &m| acc | m)
    }

    /// King attacks from sq — 4 orthogonal destinations (palace-bound checked by caller).
    pub fn king_attacks(&self, sq: u8) -> u128 {
        get_king_attacks()[sq as usize]
    }

    /// Pawn attacks from sq for the given color.
    /// Returns attack squares (forward + side if crossed river).
    pub fn pawn_attacks(&self, sq: u8, color: Color) -> u128 {
        let x = (sq % 9) as i8;
        let y = (sq / 9) as i8;
        let dir: i8 = if color == Color::Red { -1 } else { 1 };
        let mut attacks = 0u128;

        let forward_sq = (y + dir, x);
        if forward_sq.0 >= 0 && forward_sq.0 < 10 {
            attacks |= 1_u128 << (forward_sq.0 * 9 + forward_sq.1) as u8;
        }

        // Side moves only after crossing river
        let crossed = if color == Color::Red { y <= 4 } else { y >= 5 };
        if crossed {
            for dx in [-1, 1] {
                let sx = x + dx;
                if (0..9).contains(&sx) {
                    attacks |= 1_u128 << (y * 9 + sx) as u8;
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
        for dir in 0..4 {
            let ray = rays[target as usize][dir];
            let blockers = ray & occ;
            if blockers == 0 { continue; }
            let nearest = Self::lsb_index(blockers);
            if occ_color & (1_u128 << nearest) != 0
                && self.pieces[PieceType::Chariot as usize][color as usize] & (1_u128 << nearest) != 0
            {
                attackers |= 1_u128 << nearest;
            }
        }

        // Cannon attacks: needs exactly 1 screen between src and target
        for dir in 0..4 {
            let ray = rays[target as usize][dir];
            let blockers = ray & occ;
            if blockers == 0 { continue; }
            let nearest = Self::lsb_index(blockers);
            if self.pieces[PieceType::Cannon as usize][color as usize] & (1_u128 << nearest) != 0 {
                let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
                if second_blockers != 0 {
                    // Cannon can capture through its screen
                    attackers |= 1_u128 << nearest;
                }
            }
        }

        // Horse attacks: 8 L-shape destinations around target
        // Horse at SRC attacks TAR: SRC = TAR - HORSE_DELTA
        let horse_attacks_bb = self.horse_attacks(target);
        let mut horse_bb = horse_attacks_bb & occ_color;
        while horse_bb != 0 {
            let sq = Self::lsb_index(horse_bb);
            if self.pieces[PieceType::Horse as usize][color as usize] & (1_u128 << sq) != 0 {
                attackers |= 1_u128 << sq;
            }
            horse_bb &= horse_bb - 1;
        }

        // Pawn attacks: pawns that can attack target
        let pawn_attacks_bb = self.pawn_attacks(target, color);
        let mut pawn_bb = pawn_attacks_bb & occ_color;
        while pawn_bb != 0 {
            let sq = Self::lsb_index(pawn_bb);
            if self.pieces[PieceType::Pawn as usize][color as usize] & (1_u128 << sq) != 0 {
                attackers |= 1_u128 << sq;
            }
            pawn_bb &= pawn_bb - 1;
        }

        // King attacks: 4 orthogonal moves
        let king_attacks_bb = self.king_attacks(target);
        let mut king_bb = king_attacks_bb & occ_color;
        while king_bb != 0 {
            let sq = Self::lsb_index(king_bb);
            if self.pieces[PieceType::King as usize][color as usize] & (1_u128 << sq) != 0 {
                attackers |= 1_u128 << sq;
            }
            king_bb &= king_bb - 1;
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
                self.chariot_attacks(from)
            }
            _ if self.pieces[PieceType::Cannon as usize][color as usize] & (1_u128 << from) != 0 => {
                self.cannon_attacks(from)
            }
            _ if self.pieces[PieceType::Horse as usize][color as usize] & (1_u128 << from) != 0 => {
                self.horse_attacks(from)
            }
            _ if self.pieces[PieceType::Advisor as usize][color as usize] & (1_u128 << from) != 0 => {
                self.advisor_attacks(from)
            }
            _ if self.pieces[PieceType::Elephant as usize][color as usize] & (1_u128 << from) != 0 => {
                self.elephant_attacks(from)
            }
            _ if self.pieces[PieceType::King as usize][color as usize] & (1_u128 << from) != 0 => {
                self.king_attacks(from)
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
            let idx = (board2.zobrist_key as usize % moves.len());
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
}