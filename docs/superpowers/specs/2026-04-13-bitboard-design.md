# BitBoard Optimization — Design Specification

**Date:** 2026/04/13
**Author:** Claude
**Status:** Draft

---

## 1. Overview

Refactor the Chinese Chess engine from a 2D array board representation (`[[Option<Piece>; 9]; 10]`) to bitboard-based representation using the **strangler fig pattern** — bitboards are introduced alongside existing code, with subsystems migrated one at a time.

**Goals:**
- Performance: Speed up move generation, attack detection, and SEE via bitwise operations
- Memory: Smaller transposition table entries, better cache locality
- Evaluation: Faster NNUE feature plane extraction

**Non-goals:**
- No changes to search algorithm (MTDF, quiescence, etc.)
- No changes to evaluation weights
- No changes to NNUE network architecture

---

## 2. Bitboard Layout

### 2.1 Board Representation

**9×10 unpadded board** — `y=0-9, x=0-8` (standard Xiangqi board, 90 squares):

```
    x=0  x=1  x=2  x=3  x=4  x=5  x=6  x=7  x=8
y=0    0    1    2    3    4    5    6    7    8
y=1    9   10   11   12   13   14   15   16   17
y=2   18   19   20   21   22   23   24   25   26
y=3   27   28   29   30   31   32   33   34   35
y=4   36   37   38   39   40   41   42   43   44
y=5   45   46   47   48   49   50   51   52   53
y=6   54   55   56   57   58   59   60   61   62
y=7   63   64   65   66   67   68   69   70   71
y=8   72   73   74   75   76   77   78   79   80
y=9   81   82   83   84   85   86   87   88   89
```

**Bit index formula:** `sq = y * 9 + x`
**Unused bits 90-127** (top 38 bits of u128) are always 0.

All 100 squares (0-99) are real board squares. No padding bits are used in this design.

### 2.2 Bitboards Struct

```rust
pub struct Bitboards {
    /// pieces[piece_type][color] → u128 bitboard
    /// piece_type: 0=King, 1=Advisor, 2=Elephant, 3=Pawn, 4=Horse, 5=Cannon, 6=Chariot
    /// color: 0=Red, 1=Black
    pieces: [[u128; 2]; 7],
}
```

**Total storage:** 14 × 16 bytes = **224 bytes** per `Bitboards` instance.

### 2.3 Precomputed Attack Tables

```rust
/// chariot_rays[sq][dir] = u128 mask of squares in cardinal direction
/// dir: 0=North(+y), 1=South(-y), 2=East(+x), 3=West(-x)
pub static CHARIOT_RAYS: [[u128; 4]; 90] = ...;

/// cannon_screens[sq][dir] = u128 mask of squares between src and first piece
pub static CANNON_SCREENS: [[u128; 4]; 90] = ...;

/// horse_attacks[sq][horse_delta_idx] = u128 mask (usually 1 bit)
pub static HORSE_ATTACKS: [[u128; 8]; 90] = ...;

/// advisor_attacks[sq][dir] = u128 mask
pub static ADVISOR_ATTACKS: [[u128; 4]; 90] = ...;

/// elephant_attacks[sq][dir] = u128 mask
pub static ELEPHANT_ATTACKS: [[u128; 4]; 90] = ...;

/// king_attacks[sq] = u128 mask
pub static KING_ATTACKS: [u128; 90] = ...;
```

**Size of tables:**
- CHARIOT_RAYS: 90 × 4 × 16 = 5,760 bytes
- CANNON_SCREENS: 90 × 4 × 16 = 5,760 bytes
- HORSE_ATTACKS: 90 × 8 × 16 = 11,520 bytes
- ADVISOR_ATTACKS: 90 × 4 × 16 = 5,760 bytes
- ELEPHANT_ATTACKS: 90 × 4 × 16 = 5,760 bytes
- KING_ATTACKS: 90 × 16 = 1,440 bytes

**Total static tables:** ~36 KB — computed once at compile time via const initializer.

---

## 3. Migration Phases

### Phase 1 — Move Generation (Priority: HIGH)

**Deliverables:**
- `src/bitboards.rs` — New module with `Bitboards` struct and all attack tables
- Update `src/movegen.rs` to use `Bitboards` for slide attacks (chariot, cannon)
- Non-slide pieces (horse, pawn, advisor, elephant, king) continue using current logic initially
- `Bitboards::generate_pseudo_moves(color)` — Returns all pseudo-legal moves

**Validation:**
- Golden test: compare against `generate_legal_moves` for 1000 random positions

**Changes to existing code:**
- `Board` gets a `bitboards: Bitboards` field (added, not replacing `cells` yet)
- `Board::apply_action` and `Board::undo_action` update `bitboards` incrementally

### Phase 2 — Attack Detection & SEE

**Deliverables:**
- `find_least_valuable_attacker` rewritten using bitboard attackers mask
- SEE fully migrated to use Bitboards

**Validation:**
- SEE produces identical values for all positions in test suite

### Phase 3 — NNUE Input Encoding

**Deliverables:**
- `Bitboards::to_nnue_input(stm: Color) -> NNInputPlanes`
- Validated against existing `NNInputPlanes::from_board` (must match exactly)

**Validation:**
- Assert `Bitboards::to_nnue_input(board.current_side)` == `NNInputPlanes::from_board(&board)` for all test positions

### Phase 4 — Board Struct Cleanup

**Deliverables:**
- Remove `cells: [[Option<Piece>; 9]; 10]` from `Board`
- `Bitboards` becomes canonical board state
- FEN parsing/generation via `Bitboards::from_cells` and `Bitboards::to_fen`
- Remove `king_pos: RefCell` cache — bitboard `bitscan(king_bb)` is O(1)

**Validation:**
- FEN round-trip: `Board::from_fen(fen) -> to_fen() == fen` for all test FENs

---

## 4. API Reference

### 4.1 Bitboards Constructor

```rust
impl Bitboards {
    /// Create empty bitboards
    pub fn new() -> Self;

    /// Build from existing cells array (for Board::new and Board::from_fen)
    pub fn from_cells(cells: &[[Option<Piece>; 9]; 10]) -> Self;

    /// Build from FEN string
    pub fn from_fen(fen: &str) -> Self;
}
```

### 4.2 Occupancy Queries

```rust
impl Bitboards {
    /// All occupied squares for a color (union of all piece bitboards)
    pub fn occupied(&self, color: Color) -> u128;

    /// All occupied squares on the board
    pub fn occupied_all(&self) -> u128 {
        self.occupied(Color::Red) | self.occupied(Color::Black)
    }

    /// Piece at a specific square (returns None if empty)
    pub fn piece_at(&self, sq: u8) -> Option<Piece>;

    /// Piece at Coord (for Board API compatibility)
    pub fn piece_at_coord(&self, coord: Coord) -> Option<Piece> {
        self.piece_at(coord.y * 10 + coord.x)
    }
}
```

### 4.3 Attack Generation

```rust
impl Bitboards {
    /// Chariot pseudo-legal attacks from sq (all squares along 4 cardinal dirs)
    pub fn chariot_attacks(&self, sq: u8) -> u128;

    /// Cannon pseudo-legal attacks from sq (slides, captures through screen)
    pub fn cannon_attacks(&self, sq: u8) -> u128;

    /// Horse attacks from sq (8 L-shape destinations)
    pub fn horse_attacks(&self, sq: u8) -> u128;

    /// Advisor attacks from sq (4 diagonal palace-bound moves)
    pub fn advisor_attacks(&self, sq: u8) -> u128;

    /// Elephant attacks from sq (4 diagonal river-bound moves)
    pub fn elephant_attacks(&self, sq: u8) -> u128;

    /// King attacks from sq (4 orthogonal moves, palace-bound)
    pub fn king_attacks(&self, sq: u8) -> u128;

    /// Pawn attacks from sq (forward + side after river crossing)
    pub fn pawn_attacks(&self, sq: u8, color: Color) -> u128;

    /// All attackers of `color` attacking `target` square
    pub fn attackers(&self, target: u8, color: Color) -> u128;
}
```

### 4.4 Move Generation

```rust
impl Bitboards {
    /// Generate all pseudo-legal moves for a piece at `from`
    pub fn generate_moves(&self, from: u8, color: Color) -> Vec<u8>;

    /// Generate all pseudo-legal moves for a color
    pub fn generate_pseudo_moves(&self, color: Color) -> Vec<u8>;
}
```

### 4.5 Incremental Updates

```rust
impl Bitboards {
    /// Apply a move (updates bitboards in place)
    pub fn apply_move(&mut self, src: u8, dst: u8, captured: Option<Piece>);

    /// Undo a move (restores previous state)
    pub fn undo_move(&mut self, src: u8, dst: u8, captured: Option<Piece>, piece: Piece);

    /// Compute Zobrist delta for a move (without mutating)
    pub fn zobrist_delta(&self, src: u8, dst: u8, captured: Option<Piece>) -> u64;
}
```

### 4.6 FEN Serialization

```rust
impl Bitboards {
    /// Convert to FEN string
    pub fn to_fen(&self) -> String;
}
```

---

## 5. Key Implementation Details

### 5.1 Bit Index ↔ Coordinate Conversion

```rust
#[inline(always)]
pub fn sq_from_coord(x: i8, y: i8) -> u8 {
    (y * 9 + x) as u8
}

#[inline(always)]
pub fn coord_from_sq(sq: u8) -> Coord {
    Coord::new((sq % 9) as i8, (sq / 9) as i8)
}
```

### 5.2 Bit Scanning

Use compiler intrinsics for bitscan (find index of least/most significant bit):

```rust
#[inline(always)]
pub fn lsb_index(bb: u128) -> u8 {
    bb.trailing_zeros() as u8
}

#[inline(always)]
pub fn msb_index(bb: u128) -> u8 {
    127 - bb.leading_zeros() as u8
}
```

### 5.3 Chariot Attack Generation

```rust
pub fn chariot_attacks(&self, sq: u8) -> u128 {
    let occ = self.occupied_all();
    let mut attacks = u128::ZERO;
    for dir in 0..4 {
        let ray = CHARIOT_RAYS[sq as usize][dir];
        let blockers = ray & occ;
        if blockers != 0 {
            let nearest = lsb_index(blockers);
            let ray_to_nearest = CHARIOT_RAYS[sq as usize][dir] & !CHARIOT_RAYS[nearest as usize][dir];
            // Include the blocker itself if it's enemy (capture) but not own piece
            attacks |= ray_to_nearest;
        } else {
            attacks |= ray;
        }
    }
    attacks
}
```

### 5.4 Cannon Attack Generation

```rust
pub fn cannon_attacks(&self, sq: u8) -> u128 {
    let occ = self.occupied_all();
    let mut attacks = u128::ZERO;
    for dir in 0..4 {
        let ray = CHARIOT_RAYS[sq as usize][dir];
        let blockers = ray & occ;
        if blockers != 0 {
            let nearest = lsb_index(blockers);
            let second_blockers = CHARIOT_RAYS[nearest as usize][dir] & occ;
            if second_blockers != 0 {
                let second = lsb_index(second_blockers);
                let capture_ray = CHARIOT_RAYS[sq as usize][dir] & !CHARIOT_RAYS[second as usize][dir];
                attacks |= capture_ray;
            }
            // Exactly one screen: no capture squares (cannon needs screen to capture)
        } else {
            // No blockers: no captures
        }
    }
    attacks
}
```

### 5.5 Zobrist Incremental Updates

The Zobrist key update on move becomes:

```rust
pub fn apply_move(&mut self, src: u8, dst: u8, captured: Option<Piece>) {
    let zobrist = get_zobrist();
    let color = self.current_side; // tracked separately in Board

    // Remove piece from src
    let piece = self.piece_at(src).unwrap();
    self.pieces[piece.piece_type as usize][piece.color as usize] &= !(1_u128 << src);

    // Add piece at dst
    self.pieces[piece.piece_type as usize][piece.color as usize] |= 1_u128 << dst;

    // If capture: remove captured piece
    if let Some(cp) = captured {
        self.pieces[cp.piece_type as usize][cp.color as usize] &= !(1_u128 << dst);
    }
}
```

---

## 6. Test Plan

### 6.1 Phase 1 Tests

| Test | Description |
|---|---|
| `movegen_equivalence` | For 1000 random positions, `Bitboards::generate_pseudo_moves` matches `movegen::generate_legal_moves` |
| `chariot_attacks_empty` | On empty board, chariot attacks from center match precomputed rays |
| `cannon_attacks_empty` | On empty board, cannon attacks from center match precomputed screens |

### 6.2 Phase 2 Tests

| Test | Description |
|---|---|
| `see_equivalence` | For all positions, `see()` using Bitboards == original SEE values |
| `attackers_mask` | `Bitboards::attackers(sq, color)` returns correct piece squares |

### 6.3 Phase 3 Tests

| Test | Description |
|---|---|
| `nnue_equivalence` | `Bitboards::to_nnue_input()` matches `NNInputPlanes::from_board()` exactly |
| `non_king_count` | `Bitboards::count_non_king_pieces()` == original count |

### 6.4 Phase 4 Tests

| Test | Description |
|---|---|
| `fen_roundtrip` | `Board::from_fen(fen).to_fen() == fen` for all test FENs |
| `initial_position` | Bitboard initial position matches `Board::new()` |
| `zobrist_stability` | Zobrist key computed from bitboards == Board.zobrist_key |

---

## 7. File Structure Changes

```
src/
  bitboards.rs       (NEW) Bitboards struct, attack tables, move generation
  main.rs            (MOD) Add bitboards field to Board; update apply/undo actions
  movegen.rs         (MOD) Add bitboard-based movegen; keep old for comparison tests
  eval.rs            (MOD) Phase 3: update NNUE input to use bitboards
  nnue_input.rs      (MOD) Add Bitboards::to_nnue_input() method
  nn_eval.rs         (MOD) Swap NNInputPlanes source
  pgn_converter.rs   (MOD) Use bitboard FEN serialization
```

---

## 8. Open Questions

- [x] **Padding scheme:** 10×10 (2 padding columns) — RESOLVED
- [x] **Migration strategy:** Strangler fig — RESOLVED
- [x] **Implementation order:** Movegen → SEE → NNUE → Board cleanup — RESOLVED
- [ ] **Popcount builtin:** Use `u128::count_ones()` (stable Rust) vs manual loop
- [ ] **Parallel search:** Does bitboard help threads sharing state? (Probably not needed in Phase 1)

---

## 9. Success Criteria

- **Phase 1:** Benchmarks show ≥20% speedup in NPS (nodes per second) for move generation
- **Phase 2:** SEE time reduced by ≥50%
- **Phase 3:** NNUE input encoding time reduced by ≥50%
- **Phase 4:** Memory footprint reduced by ≥10% for transposition table entries
- **All phases:** 100% backward compatibility — no regressions in test suite
