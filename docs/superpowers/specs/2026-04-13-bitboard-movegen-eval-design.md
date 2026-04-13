# Bitboard Migration: movegen and eval

## Status: Draft

## Goal

Migrate `movegen` (in `src/main.rs`) and `eval_impl` (in `src/eval.rs`) to use the existing `Bitboards` infrastructure as the canonical data source, replacing all naive grid iteration (`for y in 0..10 { for x in 0..9 { board.get(Coord::new(x, y)) } }`).

## Background

### Current state

- `Board` has `bitboards: Bitboards` as canonical state; `board.get(Coord)` reads from bitboards via `piece_at_coord`
- `Board` has no `cells` array
- `Bitboards` in `src/bitboards.rs` has:
  - 14 piece-type/color bitboards (`pieces[7][2]` as `u128`)
  - Pre-computed attack tables: `CHARIOT_RAYS`, `CANNON_SCREENS`, `HORSE_ATTACKS`, `ADVISOR_ATTACKS`, `ELEPHANT_ATTACKS`, `KING_ATTACKS`
  - Methods: `chariot_attacks`, `cannon_attacks`, `horse_attacks`, `advisor_attacks`, `elephant_attacks`, `king_attacks`, `pawn_attacks`
  - `generate_moves(from, color)` — returns raw destination squares (does NOT filter own-occupied)
  - `generate_pseudo_moves(color)` — iterates `occupied(color)` via `lsb_index`, calls `generate_moves`, returns `(u8, u8, Option<Piece>)` tuples
  - `apply_move` / `undo_move` for incremental updates
  - `occupied(color)`, `occupied_all()`, `piece_at(sq)`, `piece_at_coord(Coord)`

- `movegen` module in main.rs (line 1156+) still uses grid iteration to find pieces and calls individual piece generators (`generate_pawn_moves`, `generate_chariot_moves`, etc.)
- `eval_impl` module in `src/eval.rs` (line 21+) scans the full 10×9 grid to compute material, PST, mobility, center control, etc. It re-exports `movegen` from main.rs, so the grid-scan iteration is pervasive.
- `is_capture_threat_internal` uses `movegen` functions

### Problem

The existing `Bitboards` infrastructure is not used by `movegen` or `eval`. Duplicate logic exists: grid iteration in movegen/eval, bitboard iteration already in Bitboards.

## Design

### 1. Bitboards attack methods — add color filtering

All `Bitboards` move/attack methods must filter destinations so they only return squares that are **empty or occupied by enemy pieces**. This makes them safe to use directly in move generation without additional color checks.

Methods to modify (all need `color` parameter to filter own-occupied destinations):

- **`chariot_attacks(sq, color)`** — currently returns ray_to_nearest including the blocker; filter: exclude destination if `piece_at(dst)` is own piece
- **`cannon_attacks(sq, color)`** — currently returns capture_ray including second blocker; filter: exclude if own piece at capture square
- **`horse_attacks(sq, color)`** — precomputed table returns all 8 destinations regardless of board; filter: exclude own-occupied
- **`advisor_attacks(sq, color)`** — add filter: `self.piece_at(dst).map_or(true, |p| p.color != color)`
- **`elephant_attacks(sq, color)`** — add filter: `self.piece_at(dst).map_or(true, |p| p.color != color)`
- **`king_attacks(sq, color)`** — add filter: `self.piece_at(dst).map_or(true, |p| p.color != color)`
- **`pawn_attacks(sq, color)`** — add filter: `self.piece_at(dst).map_or(true, |p| p.color != color)`

The key insight: sliding piece attacks (chariot, cannon) that currently return the blocker square itself need to check if that blocker is friendly and exclude it. Horse attacks need full filtering on all 8 destinations.

Note: `Bitboards::attackers(target, color)` computes which pieces of `color` can attack `target`. It needs the raw (pre-filter) attack functions to compute correct attack maps. Two options:
  (a) Keep raw methods separately (e.g., `chariot_attacks_raw`) and have the filtered version call it, OR
  (b) `attackers` calls the filtered version and additionally checks `piece_at(dst).color == attacker_color` — but this changes semantics.

  **Decision**: Add a `color` parameter to all attack methods. The filtered version is what callers use for move generation. `attackers` will be updated to use the filtered methods and add its own `piece.color == color` check on the result.


### 2. CORE_AREA_MASK constant

Add to `src/bitboards.rs`:

```rust
const CORE_AREA_MASK: u128 = /* mask of x∈[3,5], y∈[3,6] */;
// = sum of (1_u128 << (y*9 + x)) for y in 3..=6, x in 3..=5
```

### 3. New `movegen_bb` module in main.rs

Located alongside existing `movegen`, not replacing it until validated.

#### `generate_pseudo_moves(board, &Board, color: Color) -> SmallVec<[Action; 32]>`

```rust
pub fn generate_pseudo_moves(board: &Board, color: Color) -> SmallVec<[Action; 32]> {
    let mut moves = SmallVec::new();
    let bitboards = &board.bitboards;
    let mut occ = bitboards.occupied(color);

    while occ != 0 {
        let from_sq = Bitboards::lsb_index(occ);
        let from_coord = coord_from_sq(from_sq);
        let piece = bitboards.piece_at(from_sq).unwrap();

        for dst_sq in bitboards.generate_moves(from_sq, color) {
            let tar_coord = coord_from_sq(dst_sq);
            let captured = bitboards.piece_at(dst_sq); // None if empty
            moves.push(Action::new(from_coord, tar_coord, captured));
        }

        occ &= occ - 1;
    }
    moves
}
```

Note: `Bitboards::generate_moves` now does its own own-occupied filtering (from step 1), so `captured` correctly distinguishes captures from quiets.

#### `generate_capture_moves(board, &Board, color: Color) -> SmallVec<[Action; 32]>`

```rust
pub fn generate_capture_moves(board: &Board, color: Color) -> SmallVec<[Action; 32]> {
    generate_pseudo_moves(board, color).into_iter()
        .filter(|a| a.captured.is_some())
        .collect()
}
```

#### `generate_legal_moves(board, &mut Board, color: Color) -> SmallVec<[Action; 32]>`

Same as existing: calls `generate_pseudo_moves`, then `is_legal_move` for each, returns legal ones with `is_check` set.

#### `is_capture_threat_internal(board, &Board, attacker_color: Color) -> bool`

```rust
fn is_capture_threat_internal(board: &Board, attacker_color: Color) -> bool {
    let bitboards = &board.bitboards;
    let mut occ = bitboards.occupied(attacker_color);

    while occ != 0 {
        let from_sq = Bitboards::lsb_index(occ);
        for dst_sq in bitboards.generate_moves(from_sq, attacker_color) {
            if let Some(p) = bitboards.piece_at(dst_sq) {
                if p.color == attacker_color.opponent() && p.piece_type != PieceType::King {
                    return true;
                }
            }
        }
        occ &= occ - 1;
    }
    false
}
```

### 4. Eval migration

All functions in `eval_impl` replace grid iteration with bitboard iteration.

#### `game_phase(board: &Board) -> i32`

```rust
pub fn game_phase(board: &Board) -> i32 {
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
```

#### `center_control(board: &Board, color: Color, phase: i32) -> i32`

Replace geometric checks with bitboard operations:

```rust
fn center_control(board: &Board, color: Color, phase: i32) -> i32 {
    let occ_all = board.bitboards.occupied_all();
    let our_chariots = board.bitboards.piece_bitboard(PieceType::Chariot, color);
    let our_cannons = board.bitboards.piece_bitboard(PieceType::Cannon, color);
    let our_horses = board.bitboards.piece_bitboard(PieceType::Horse, color);
    let our_pawns = board.bitboards.piece_bitboard(PieceType::Pawn, color);

    let core = CORE_AREA_MASK;

    let our_attacks = {
        let mut a = 0i32;
        // Chariot/cannon: attack = chariot_attacks(sq) | cannon_attacks(sq) & CORE
        let mut bb = our_chariots;
        while bb != 0 {
            let sq = Bitboards::lsb_index(bb);
            a += (board.bitboards.chariot_attacks(sq) & core).count_ones() as i32 * 2;
            bb &= bb - 1;
        }
        let mut bb = our_cannons;
        while bb != 0 {
            let sq = Bitboards::lsb_index(bb);
            a += (board.bitboards.cannon_attacks(sq) & core).count_ones() as i32 * 2;
            bb &= bb - 1;
        }
        let mut bb = our_horses;
        while bb != 0 {
            let sq = Bitboards::lsb_index(bb);
            a += (board.bitboards.horse_attacks(sq) & core).count_ones() as i32;
            bb &= bb - 1;
        }
        let mut bb = our_pawns;
        while bb != 0 {
            let sq = Bitboards::lsb_index(bb);
            a += (board.bitboards.pawn_attacks(sq, color) & core).count_ones() as i32;
            bb &= bb - 1;
        }
        a
    };

    // Same for opponent, subtract
    let opp_attacks = /* same logic for opponent color */;
    let net_attacks = (our_attacks - opp_attacks) * color.sign();
    net_attacks * 5 * (30 + 70 * phase) / 100
}
```

#### `horse_mobility(board: &Board, pos: Coord, color: Color) -> i32`

After `horse_attacks(sq, color)` is updated to filter own-occupied destinations, we need to account for knee checks. The precomputed attack table gives all 8 L-destinations regardless of knee state. For true mobility (count of legal jumps), we still need to check the knee board state.

The current implementation iterates `HORSE_DELTAS` and checks `board.get(block).is_none()`. Since `horse_attacks` is now filtered and the knee check is the only remaining board-dependent filter, we need to keep the knee check in `horse_mobility`:

```rust
fn horse_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
    let mut mobility = 0;
    for i in 0..8 {
        let (dx, dy) = HORSE_DELTAS[i];
        let (bx, by) = HORSE_BLOCKS[i];
        let tar = Coord::new(pos.x + dx, pos.y + dy);
        let block = Coord::new(pos.x + bx, pos.y + by);
        // Valid jump: target is on board AND horse head is empty AND target not own-occupied
        if tar.is_valid() && board.get(block).is_none() {
            let sq = (tar.y * 9 + tar.x) as u8;
            // Check own-occupied: horse_attacks now filters, but we need to check knee too
            // Since horse_attacks returns all 8 destinations without knee check,
            // we count any destination where knee is empty and target not own-occupied
            if board.get(tar).is_none_or(|p| p.color != color) {
                mobility += 1;
            }
        }
    }
    mobility * 10
}
```

Note: `horse_attacks(sq, color)` itself no longer checks knee state (knee is a precondition, not part of the attack destination filtering). So `horse_mobility` must retain the knee check.

Let me re-state the eval migration more clearly:

#### Functions that do full-board grid scan → migrate to bitboard iteration:

- `game_phase` → `occupied_all()` + lsb loop
- `center_control` → `piece_bitboard` + attack masks + `CORE_AREA_MASK`
- `attack_rewards` → `occupied(color)` + `occupied(opponent)` + attack methods
- `piece_coordination` → bitboard iteration over own pieces
- `pawn_structure` → `piece_bitboard(Pawn, color)` + lsb loop
- `elephant_structure` → `piece_bitboard(Elephant, color)` + lsb loop  
- `king_safety` → `piece_bitboard(King, color)` + attack iteration
- `handcrafted_evaluate` → replace grid scan with `occupied_all()` + lsb loop, call per-piece helpers

#### Functions that only iterate small constant-size sets → keep as-is:

- `horse_mobility` — 8 horse offsets only
- `cannon_support` — 4 directions only
- `chariot_mobility` — 4 directions only

#### `attack_rewards` details

Currently scans all squares, builds SmallVec of pieces, then checks attack/defense relationships. With bitboards:
- Iterate `occupied(color)` → `our_pieces`
- Iterate `occupied(opponent)` → `enemy_pieces`  
- For each enemy piece, check `attackers(dst_sq, color)` via `Bitboards::attackers`

#### `piece_coordination` details

Currently iterates all pieces to find horse+chariot presence and positions. Replace with bitboard iteration over own pieces.

### 5. Migration order

1. Modify `Bitboards` attack methods to filter own-occupied squares (step 1)
2. Add `CORE_AREA_MASK` constant (step 2)
3. Write `movegen_bb` module (step 3) — validate against existing tests
4. Migrate eval functions (step 4)
5. Once all tests pass, replace `movegen` calls with `movegen_bb` in callers
6. Remove old `movegen` module (or keep as fallback if needed)

### 6. Validation

- `cargo test` baseline before changes
- After modifying Bitboards methods: run existing `bitboards` tests
- After writing `movegen_bb`: compare `generate_pseudo_moves` output on initial position and after random move sequences
- After eval migration: `handcrafted_evaluate` scores must be identical before/after (within integer rounding)

## Open questions

1. `chariot_mobility` and `cannon_support` — currently iterate `DIRS_4` and check paths. After chariot_attacks/cannon_attacks add color filtering, do these functions need to stay as-is with their explicit path checking, or can they simplify to use the filtered attack masks? (chariot_mobility counts empty squares in 4 directions; cannon_support counts platform pieces)
2. `attackers(target, color)` — currently uses raw attack functions. After adding `color` to all attack methods, should `attackers` use the filtered versions (and add `piece.color == color` check on the result), or should it keep raw methods separate?