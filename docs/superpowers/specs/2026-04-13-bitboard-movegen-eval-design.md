# Bitboard Migration: movegen → bitboards

## Status: Draft

## Goal

Migrate the entire `movegen` module logic from `src/main.rs` into `src/bitboards.rs`, replacing the current `movegen::generate_*_moves(board, pos, color)` grid-scan approach with bitboard-first implementations. After migration, `movegen` in main.rs calls into bitboards; then the movegen module can be removed or simplified.

## Background

### Current state

**Board canonical state:** `Board` has `bitboards: Bitboards` as canonical; `board.get(Coord)` reads from bitboards via `piece_at_coord`. No `cells` array.

**Bitboards** (`src/bitboards.rs`):
- `pieces[7][2]` as `u128` — 7 piece types × 2 colors
- Pre-computed attack tables: `CHARIOT_RAYS`, `CANNON_SCREENS`, `HORSE_ATTACKS`, `ADVISOR_ATTACKS`, `ELEPHANT_ATTACKS`, `KING_ATTACKS` (all OnceLock lazy init)
- Attack methods: `chariot_attacks(sq)`, `cannon_attacks(sq)`, `horse_attacks(sq)`, `advisor_attacks(sq)`, `elephant_attacks(sq)`, `king_attacks(sq)`, `pawn_attacks(sq, color)`
- `generate_moves(from, color)` — calls piece-type attack methods; returns `SmallVec<[u8; 17]>` of destination square indices (no color filtering)
- `generate_pseudo_moves(color)` — iterates `occupied(color)` via `lsb_index`; returns `SmallVec<[(u8, u8, Option<Piece>); 64]>`
- `attackers(target, color)` — uses raw attack functions; returns `u128` mask of attackers
- `apply_move`/`undo_move` for incremental updates
- Helper: `lsb_index(bb)`, `coord_from_sq(sq)`, `sq_from_coord(x, y)`

**Movegen** (`src/main.rs`, `pub mod movegen {}`):
- `generate_pawn_moves(board, pos, color)` → `MoveBuf` (Coord destinations)
- `generate_horse_moves(board, pos, color)` → `MoveBuf` (knee checked)
- `generate_chariot_moves(board, pos, color)` → `MoveBuf`
- `generate_cannon_moves(board, pos, color)` → `MoveBuf`
- `generate_elephant_moves(board, pos, color)` → `MoveBuf` (eye square checked, river bound)
- `generate_advisor_moves(board, pos, color)` → `MoveBuf` (palace bound)
- `generate_king_moves(board, pos, color)` → `MoveBuf` (palace bound + face-to-face)
- `generate_pseudo_moves(board, color)` → `SmallVec<[Action; 32]>` (grid scan)
- `generate_legal_moves(board, color)` → `SmallVec<[Action; 32]>`
- `generate_capture_moves(board, color)` → `SmallVec<[Action; 32]>`
- `is_legal_move(board, action, color)` → `(bool, bool)`
- `see(board, src, tar)` → `i32`
- `find_least_valuable_attacker(board, tar, side)` → `(Option<Coord>, i32)` (uses `attackers()` bitboard)

**Eval** (`src/eval.rs`, `pub mod eval_impl {}`):
- `handcrafted_evaluate(board, side, initiative)` → `i32` (top-level, grid scan)
- Per-component functions: `game_phase`, `center_control`, `attack_rewards`, `piece_coordination`, `pawn_structure`, `elephant_structure`, `king_safety`, `horse_mobility`, `chariot_mobility`, `cannon_support`

### Problems

1. All movegen piece-type functions use grid iteration (`for y in 0..10, for x in 0..9`)
2. All eval functions use grid iteration
3. Bitboards attack methods don't filter own-occupied destinations
4. `horse_attacks` doesn't check knee (intermediate square) emptiness
5. Sliding attacks (chariot/cannon) return the blocker square itself when friendly pieces block the ray — needs filtering

## Design

### 1. Bitboards attack methods — complete rewrite

All attack/move methods take `color: Color` and filter destinations to empty or enemy-occupied only. Horse additionally checks knee squares.

**Helper constant needed:** `HORSE_BLOCKS` and `HORSE_DELTAS` must be accessible from bitboards.rs. Currently defined in main.rs. Options:
- Move them to a shared location (e.g., a `constants` module)
- Pass them as function parameters
- Duplicate the constant definitions in bitboards.rs

Decision: **Duplicate constants in bitboards.rs** — keeps the module self-contained, no cross-module dependencies for static data.

**`chariot_attacks(sq: u8, color: Color) -> u128`**

Returns ALL reachable destination squares (empty OR enemy-capturable) in 4 cardinal directions. Friendly pieces block the ray — they are not capturable destinations.

```rust
pub fn chariot_attacks(&self, sq: u8, color: Color) -> u128 {
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);
    let rays = get_chariot_rays();
    let mut attacks = 0u128;

    for dir in 0..4 {
        let ray = rays[sq as usize][dir];
        let blockers = ray & occ;
        if blockers == 0 {
            attacks |= ray;  // Clear path — all squares reachable
        } else {
            let nearest = Self::lsb_index(blockers);
            // Only include squares up to nearest if it is NOT our piece
            if occ_color & (1_u128 << nearest) == 0 {
                let ray_to_nearest = ray & !(rays[nearest as usize][dir]);
                attacks |= ray_to_nearest;  // Includes the nearest enemy square (capture)
            }
            // If nearest is our own piece: no squares attacked in this direction (blocked)
        }
    }
    attacks
}
```

**`cannon_attacks(sq: u8, color: Color) -> u128`**

Returns all reachable destination squares (empty AND enemy-capturable) in 4 cardinal directions. Cannon slides: it can move to any empty square before the first screen piece, or capture exactly the enemy piece beyond one screen. Friendly pieces are excluded from both quiet and capture destinations.

```rust
pub fn cannon_attacks(&self, sq: u8, color: Color) -> u128 {
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);
    let rays = get_chariot_rays();
    let mut attacks = 0u128;

    for dir in 0..4 {
        let ray = rays[sq as usize][dir];
        let blockers = ray & occ;
        if blockers == 0 {
            attacks |= ray;  // Clear path — all empty squares reachable
        } else {
            let nearest = Self::lsb_index(blockers);  // First screen piece
            // Quiet moves: all empty squares before (and not including) the screen
            attacks |= ray & !(rays[nearest as usize][dir]);

            let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
            if second_blockers != 0 {
                let second = Self::lsb_index(second_blockers);  // Capture target
                // Capture only if target is NOT our own piece
                if occ_color & (1_u128 << second) == 0 {
                    attacks |= 1_u128 << second;  // Capture = landing on the target square
                }
            }
        }
    }
    attacks
}
```

**`horse_attacks(sq: u8, color: Color) -> u128`**

Horse needs knee checking. The precomputed table gives destination squares; we need to filter by both board occupancy (same as other pieces) AND knee emptiness (new requirement). Since the knee is a precondition that requires board state, we can't use a pure precomputed mask — we must check `occupied_all()` for knee squares at runtime.

```rust
pub fn horse_attacks(&self, sq: u8, color: Color) -> u128 {
    let x = (sq % 9) as i8;
    let y = (sq / 9) as i8;
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);

    // HORSE_DELTAS[i] = (target_dx, target_dy)
    // HORSE_BLOCKS[i] = (knee_dx, knee_dy) relative to horse position
    let horse_deltas: [(i8, i8); 8] = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)];
    let horse_blocks: [(i8, i8); 8] = [(1, 0), (1, 0), (-1, 0), (-1, 0), (0, 1), (0, -1), (0, 1), (0, -1)];

    let mut attacks = 0u128;
    for i in 0..8 {
        let (dx, dy) = horse_deltas[i];
        let (bx, by) = horse_blocks[i];
        let tar_x = x + dx;
        let tar_y = y + dy;
        let knee_x = x + bx;
        let knee_y = y + by;

        // Knee must be empty
        if (0..9).contains(&knee_x) && (0..10).contains(&knee_y) {
            let knee_sq = (knee_y * 9 + knee_x) as u8;
            if occ & (1_u128 << knee_sq) != 0 {
                continue; // knee is blocked
            }
        } else {
            continue; // knee off board
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
```

**`advisor_attacks(sq: u8, color: Color) -> u128`**

```rust
pub fn advisor_attacks(&self, sq: u8, color: Color) -> u128 {
    let x = (sq % 9) as i8;
    let y = (sq / 9) as i8;
    let occ_color = self.occupied(color);

    // 4 diagonal destinations, palace-bound
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
        // Filter own-occupied
        if occ_color & (1_u128 << tsq) == 0 {
            attacks |= 1_u128 << tsq;
        }
    }
    attacks
}
```

Note: `Coord::in_palace(color)` is defined in main.rs and imported into bitboards.rs via the `use crate::` re-exports at the top of bitboards.rs.

**`elephant_attacks(sq: u8, color: Color) -> u128`**

```rust
pub fn elephant_attacks(&self, sq: u8, color: Color) -> u128 {
    let x = (sq % 9) as i8;
    let y = (sq / 9) as i8;
    let occ_color = self.occupied(color);

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
            if self.occupied_all() & (1_u128 << eye_sq) != 0 {
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
```

**`king_attacks(sq: u8, color: Color) -> u128`**

```rust
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
        // Filter own-occupied
        if occ_color & (1_u128 << tsq) == 0 {
            attacks |= 1_u128 << tsq;
        }
    }
    attacks
}
```

**`pawn_attacks(sq: u8, color: Color) -> u128`**

```rust
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
```

### 2. `generate_moves(from: u8, color: Color) -> SmallVec<[u8; 17]>`

Update to use new color-filtered attack methods:

```rust
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
        bb &= bb - 1;
    }
    destinations
}
```

### 3. `attackers(target: u8, color: Color) -> u128`

Update to use the new color-filtered attack methods. Since each attack method now filters own-occupied destinations, `attackers` needs to check if a piece actually can reach the target. Using the filtered attack maps plus a color check on the result:

```rust
pub fn attackers(&self, target: u8, color: Color) -> u128 {
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);
    let rays = get_chariot_rays();
    let mut attackers = 0u128;

    // Chariot: find nearest piece in each direction, check if it's a chariot of color
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

    // Cannon: needs exactly 1 screen between
    for dir in 0..4 {
        let ray = rays[target as usize][dir];
        let blockers = ray & occ;
        if blockers == 0 { continue; }
        let nearest = Self::lsb_index(blockers);
        if self.pieces[PieceType::Cannon as usize][color as usize] & (1_u128 << nearest) != 0 {
            let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
            if second_blockers != 0 {
                attackers |= 1_u128 << nearest;
            }
        }
    }

    // Horse: use horse_attacks(target, color) to find squares that can jump to target
    // horse_attacks(target) returns squares where a horse CAN jump TO target
    let horse_attacks_bb = self.horse_attacks(target, color);
    let mut horse_bb = horse_attacks_bb & occ_color;
    while horse_bb != 0 {
        let sq = Self::lsb_index(horse_bb);
        if self.pieces[PieceType::Horse as usize][color as usize] & (1_u128 << sq) != 0 {
            attackers |= 1_u128 << sq;
        }
        horse_bb &= horse_bb - 1;
    }

    // Pawn: use pawn_attacks(target, color) — if target is in our pawn's attack set, we attack
    let pawn_attacks_bb = self.pawn_attacks(target, color);
    let mut pawn_bb = pawn_attacks_bb & occ_color;
    while pawn_bb != 0 {
        let sq = Self::lsb_index(pawn_bb);
        if self.pieces[PieceType::Pawn as usize][color as usize] & (1_u128 << sq) != 0 {
            attackers |= 1_u128 << sq;
        }
        pawn_bb &= pawn_bb - 1;
    }

    // King: adjacent orthogonal
    let king_attacks_bb = self.king_attacks(target, color);
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
```

Note: `horse_attacks(target, color)` already filters own-occupied, so `horse_attacks_bb & occ_color` is the set of our horses that can attack target. But we additionally need to verify it's actually a horse (the `pieces[Horse][color]` check).

### 4. New movegen functions in bitboards.rs

These mirror the current main.rs movegen functions but operate on bitboards:

**`generate_pseudo_moves_bitboards(board: &Board, color: Color) -> SmallVec<[Action; 32]>`**

```rust
pub fn generate_pseudo_moves_bitboards(board: &Board, color: Color) -> SmallVec<[Action; 32]> {
    let mut moves = SmallVec::new();
    let bitboards = &board.bitboards;
    let mut occ = bitboards.occupied(color);

    while occ != 0 {
        let from_sq = Bitboards::lsb_index(occ);
        let from_coord = coord_from_sq(from_sq);
        // piece is guaranteed to exist since we're iterating occupied squares
        let piece = bitboards.piece_at(from_sq).unwrap();

        for dst_sq in bitboards.generate_moves(from_sq, color) {
            let tar_coord = coord_from_sq(dst_sq);
            let captured = bitboards.piece_at(dst_sq); // None if empty, Some(enemy) if capture
            moves.push(Action::new(from_coord, tar_coord, captured));
        }

        occ &= occ - 1;
    }
    moves
}
```

### 5. Update `movegen` module in main.rs

After bitboards methods are correct, update `movegen` to delegate to bitboards:

**`generate_pseudo_moves`** — replace grid scan with `bitboards.generate_pseudo_moves_bitboards()`

**`generate_capture_moves`** — filter captures from pseudo moves

**`generate_legal_moves`** — unchanged (calls `is_legal_move` which uses `board.bitboards.apply_move`)

**`is_legal_move`** — unchanged

**`see`** — unchanged (uses `find_least_valuable_attacker` which uses `attackers()`)

**`find_least_valuable_attacker`** — unchanged (already uses `board.bitboards.attackers()`)

**`is_capture_threat_internal`** — replace grid scan with bitboard iteration

### 6. Eval migration

After movegen is stable, migrate `eval_impl` functions. Full-grid scans → bitboard iteration:
- `game_phase` → `occupied_all()` + lsb_index loop
- `center_control` → piece bitboards + attack masks (with `CORE_AREA_MASK`)
- `attack_rewards` → `occupied(color)` / `occupied(opponent)` + attack methods
- `piece_coordination` → bitboard iteration
- `pawn_structure` → `piece_bitboard(Pawn, color)` + lsb loop
- `elephant_structure` → `piece_bitboard(Elephant, color)` + lsb loop  
- `king_safety` → `piece_bitboard(King, color)` + attack iteration
- `handcrafted_evaluate` → `occupied_all()` + lsb loop

Mobility functions (`horse_mobility`, `cannon_support`, `chariot_mobility`) operate on a single piece's local neighborhood only — they can stay with small constant-time loops, or be simplified using bitboard attack counts.

### 7. CORE_AREA_MASK constant

```rust
/// Mask of core area squares: x∈[3,5], y∈[3,6]
pub const CORE_AREA_MASK: u128 = {
    let mut mask = 0u128;
    let mut sq = 0u8;
    while sq < BOARD_SQ_COUNT as u8 {
        let x = sq % 9;
        let y = sq / 9;
        if x >= 3 && x <= 5 && y >= 3 && y <= 6 {
            mask |= 1_u128 << sq;
        }
        sq += 1;
    }
    mask
};
```

### 8. King face-to-face check

Current `generate_king_moves` checks the face-to-face rule (kings cannot be on the same file with no pieces between). This rule is **not** encoded in `king_attacks` — it must be checked in `generate_moves` or in the king move generator.

After migration: `generate_moves` for the king still needs to filter destinations based on the face-to-face rule. Since `king_attacks` doesn't know about this rule, the filtering happens in `generate_moves` when called from `generate_pseudo_moves`. Specifically, after getting raw king attack destinations, filter out moves that would place the king on a square where it faces the enemy king with no pieces between.

This is already handled by the existing `is_legal_move` check — after a provisional move is made, `board.is_check(side)` will detect if the king is in check from face-to-face. So no change to king move generation is needed beyond what `is_legal_move` already does.

### 9. Elephant river check

`elephant_attacks` checks `target_coord.crosses_river(color)` — this remains unchanged.

### 10. Advisor and King palace checks

`advisor_attacks` now checks `target.in_palace(color)` for each diagonal destination — palace bounds are enforced by the method itself.

`king_attacks` now checks `target.in_palace(color)` for each orthogonal destination — palace bounds are enforced by the method itself.

**Note on face-to-face rule**: The king face-to-face rule (kings cannot be on the same file with no pieces between) is **not** encoded in `king_attacks`. This rule requires board-wide information (checking the entire file for pieces) which cannot be determined from a single king's attack map alone. The rule is enforced by `is_legal_move`: after a provisional king move is made, `board.is_check(side)` detects if the king is left in check from face-to-face.

## Migration order

1. **Step 1**: Rewrite all `Bitboards` attack methods with color filtering + horse knee check
2. **Step 2**: Update `attackers` to use new filtered methods  
3. **Step 3**: Update `generate_moves` to use new filtered methods
4. **Step 4**: Add `generate_pseudo_moves_bitboards` function
5. **Step 5**: Add `CORE_AREA_MASK` constant
6. **Step 6**: Update `movegen::generate_pseudo_moves` to delegate to bitboards
7. **Step 7**: Update `movegen::is_capture_threat_internal` to use bitboards
8. **Step 8**: Run tests — validate movegen produces identical output
9. **Step 9**: Migrate eval functions (game_phase, center_control, attack_rewards, etc.)
10. **Step 10**: Full test suite validation

## Validation

- `cargo test` baseline before changes
- After step 1-3: run bitboards module tests
- After step 8: `generate_pseudo_moves` output matches on initial position and after random move sequences
- After step 10: `handcrafted_evaluate` scores identical before/after (within int rounding)