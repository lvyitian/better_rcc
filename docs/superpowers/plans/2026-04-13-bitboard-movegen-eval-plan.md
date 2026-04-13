# Bitboard Movegen Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite all Bitboards attack methods with color filtering + knee check, then migrate movegen to use bitboards, then migrate eval.

**Architecture:** Attack methods in `src/bitboards.rs` take `color` parameter and filter own-occupied destinations. Horse attacks additionally check knee (intermediate square) emptiness using board occupancy. Movegen in `src/main.rs` delegates to bitboards. Eval migrates grid scans to bitboard iteration.

**Tech Stack:** Rust, smallvec, OnceLock for lazy static init

---

## File Structure

- **Modify**: `src/bitboards.rs` — rewrite attack methods, add `generate_pseudo_moves_bitboards`, add `CORE_AREA_MASK`
- **Modify**: `src/main.rs` — update `movegen` to delegate to bitboards
- **Modify**: `src/eval.rs` — migrate eval_impl grid scans to bitboard iteration
- **Test**: Tests live inline in `src/bitboards.rs` (#[cfg(test)] mod tests) and `src/main.rs`

---

## Task 1: Rewrite `chariot_attacks` with color filter

**Files:**
- Modify: `src/bitboards.rs:335-352` (replace existing `chariot_attacks`)

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:335-352` to see the existing method.

- [ ] **Step 2: Replace with color-filtered version**

Replace the existing `chariot_attacks` with:

```rust
/// Chariot attacks from sq — all squares in 4 cardinal directions until first blocker.
/// Returns all reachable squares (empty OR enemy-capturable). Friendly pieces block.
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
                attacks |= ray_to_nearest;  // Includes nearest enemy square (capture)
            }
            // If nearest is our own piece: no squares attacked in this direction
        }
    }
    attacks
}
```

- [ ] **Step 3: Run bitboards tests to verify no regression**

Run: `cargo test bitboards::`
Expected: All bitboards tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add color filter to chariot_attacks"
```

---

## Task 2: Rewrite `cannon_attacks` with color filter

**Files:**
- Modify: `src/bitboards.rs:355-376`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:355-376`.

- [ ] **Step 2: Replace with color-filtered version**

```rust
/// Cannon attacks from sq — slides, returns empty squares and captures (enemy beyond 1 screen).
/// Excludes own-occupied squares from both quiet and capture destinations.
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
            let nearest = Self::lsb_index(blockers);
            // Quiet moves: all empty squares before (not including) the screen
            attacks |= ray & !(rays[nearest as usize][dir]);

            let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
            if second_blockers != 0 {
                let second = Self::lsb_index(second_blockers);
                // Capture only if target is NOT our own piece
                if occ_color & (1_u128 << second) == 0 {
                    attacks |= 1_u128 << second;  // Capture = landing on target square
                }
            }
        }
    }
    attacks
}
```

- [ ] **Step 3: Run bitboards tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add color filter to cannon_attacks"
```

---

## Task 3: Rewrite `horse_attacks` with color filter + knee check

**Files:**
- Modify: `src/bitboards.rs:378-381` (replace existing), add HORSE_DELTAS/HORSE_BLOCKS constants inside function

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:378-381`.

- [ ] **Step 2: Replace with knee-check + color-filtered version**

```rust
/// Horse attacks from sq — 8 L-shape destinations, knee square must be empty.
/// Filters out own-occupied destination squares.
pub fn horse_attacks(&self, sq: u8, color: Color) -> u128 {
    let x = (sq % 9) as i8;
    let y = (sq / 9) as i8;
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);

    // Horse deltas and knee positions (same as main.rs constants)
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
```

- [ ] **Step 3: Run bitboards tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: rewrite horse_attacks with knee check and color filter"
```

---

## Task 4: Rewrite `advisor_attacks` with color filter

**Files:**
- Modify: `src/bitboards.rs:383-386`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:383-386`.

- [ ] **Step 2: Replace with palace-checked + color-filtered version**

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add color filter to advisor_attacks"
```

---

## Task 5: Rewrite `elephant_attacks` with color filter + eye check + river check

**Files:**
- Modify: `src/bitboards.rs:388-391`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:388-391`.

- [ ] **Step 2: Replace with full-filtered version**

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: rewrite elephant_attacks with eye check, river check, and color filter"
```

---

## Task 6: Rewrite `king_attacks` with color filter

**Files:**
- Modify: `src/bitboards.rs:393-396`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:393-396`.

- [ ] **Step 2: Replace with palace-checked + color-filtered version**

```rust
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add color filter to king_attacks"
```

---

## Task 7: Rewrite `pawn_attacks` with color filter

**Files:**
- Modify: `src/bitboards.rs:400-422`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:400-422`.

- [ ] **Step 2: Replace with color-filtered version**

```rust
/// Pawn attacks from sq for the given color.
/// Returns attack squares (forward + side if crossed river).
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
```

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add color filter to pawn_attacks"
```

---

## Task 8: Update `generate_moves` to use color-filtered methods

**Files:**
- Modify: `src/bitboards.rs:501-535`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:501-535`.

- [ ] **Step 2: Update calls to pass color parameter**

Change each attack method call to pass `color`:
- `self.chariot_attacks(from)` → `self.chariot_attacks(from, color)`
- `self.cannon_attacks(from)` → `self.cannon_attacks(from, color)`
- `self.horse_attacks(from)` → `self.horse_attacks(from, color)`
- `self.advisor_attacks(from)` → `self.advisor_attacks(from, color)`
- `self.elephant_attacks(from)` → `self.elephant_attacks(from, color)`
- `self.king_attacks(from)` → `self.king_attacks(from, color)`

The rest of `generate_moves` (extracting destinations from attack bitboard) stays the same.

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: pass color to all attack methods in generate_moves"
```

---

## Task 9: Update `attackers` to use new filtered methods

**Files:**
- Modify: `src/bitboards.rs:427-496`

- [ ] **Step 1: Read current implementation**

Read `src/bitboards.rs:427-496`.

- [ ] **Step 2: Update method calls**

All calls to `horse_attacks(target)` → `horse_attacks(target, color)`. The rest stays the same since the attack methods themselves now do the filtering.

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: pass color to horse_attacks in attackers method"
```

---

## Task 10: Add `CORE_AREA_MASK` constant

**Files:**
- Modify: `src/bitboards.rs` (add near top of module, after `BOARD_SQ_COUNT`)

- [ ] **Step 1: Read current constants section**

Read `src/bitboards.rs:14` (BOARD_SQ_COUNT) to understand placement.

- [ ] **Step 2: Add constant**

Add after `pub const BOARD_SQ_COUNT: usize = 90;`:

```rust
/// Mask of core area squares: x∈[3,5], y∈[3,6] (x=3,4,5; y=3,4,5,6)
pub const CORE_AREA_MASK: u128 = {
    let mut mask = 0u128;
    let mut y = 3;
    while y <= 6 {
        let mut x = 3;
        while x <= 5 {
            let sq = (y * 9 + x) as u8;
            mask |= 1_u128 << sq;
            x += 1;
        }
        y += 1;
    }
    mask
};
```

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add CORE_AREA_MASK constant"
```

---

## Task 11: Add `generate_pseudo_moves_bitboards` function

**Files:**
- Modify: `src/bitboards.rs` (add after `generate_moves` method, around line 540)

- [ ] **Step 1: Read current `generate_pseudo_moves` in bitboards**

Read `src/bitboards.rs:539-554` to see the existing function (which returns `(u8, u8, Option<Piece>)` tuples).

- [ ] **Step 2: Add new function that returns Action structs**

Add after the existing `generate_pseudo_moves`:

```rust
/// Generate all pseudo-legal moves for a color, returning Action structs.
/// Iterates occupied squares via bitboards, calls generate_moves for destinations.
pub fn generate_pseudo_moves_bitboards(board: &Board, color: Color) -> SmallVec<[Action; 32]> {
    let mut moves = SmallVec::new();
    let bitboards = &board.bitboards;
    let mut occ = bitboards.occupied(color);

    while occ != 0 {
        let from_sq = Self::lsb_index(occ);
        let from_coord = coord_from_sq(from_sq);
        // piece is guaranteed to exist since we're iterating occupied squares
        let _piece = bitboards.piece_at(from_sq).unwrap();

        for dst_sq in bitboards.generate_moves(from_sq, color) {
            let tar_coord = coord_from_sq(dst_sq);
            let captured = bitboards.piece_at(dst_sq); // None if empty, Some(enemy) if capture
            moves.push(Action::new(from_coord, tar_coord, captured));
        }

        occ &= occ - 1; // clear LSB
    }
    moves
}
```

Note: `Action` and `Board` must be accessible from bitboards.rs. Currently `bitboards.rs` re-exports from `crate` at the top. Verify `Action` is pub in main.rs and accessible.

- [ ] **Step 3: Run tests**

Run: `cargo test bitboards::`
Expected: All PASS (may need to add import if `Action` isn't in scope)

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "bitboards: add generate_pseudo_moves_bitboards function"
```

---

## Task 12: Migrate `movegen::generate_pseudo_moves` to use bitboards

**Files:**
- Modify: `src/main.rs:1341-1367` (replace grid scan body)

- [ ] **Step 1: Read current `generate_pseudo_moves` in movegen**

Read `src/main.rs:1341-1367`.

- [ ] **Step 2: Replace grid scan with delegation to bitboards**

Replace the `for y in 0..10` grid scan body with:

```rust
// Delegate to bitboards for pseudo-legal moves
bitboards::generate_pseudo_moves_bitmaps(board, color)
```

Wait — `generate_pseudo_moves_bitboards` returns `SmallVec<[Action; 32]>`, which is exactly what movegen needs. But we need to call it via the bitboards instance. Since `Action` is the same type and the function signature matches, we can replace the body directly.

Replace the body of `generate_pseudo_moves` (lines 1342-1366) with:

```rust
bitboards::generate_pseudo_moves_bitboards(board, color)
```

This requires ensuring `bitboards` module is accessible. Since `bitboards` is `pub mod bitboards` at crate root, it should be accessible as `crate::bitboards::`.

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "movegen: delegate generate_pseudo_moves to bitboards"
```

---

## Task 13: Migrate `movegen::is_capture_threat_internal` to use bitboards

**Files:**
- Modify: `src/main.rs:2116-2145`

- [ ] **Step 1: Read current implementation**

Read `src/main.rs:2116-2145`.

- [ ] **Step 2: Replace grid scan with bitboard iteration**

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

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "movegen: migrate is_capture_threat_internal to bitboards"
```

---

## Task 14: Validate movegen output equivalence

**Files:**
- Test: `src/bitboards.rs` test module — add comparison test

- [ ] **Step 1: Write equivalence test**

In `src/bitboards.rs` test module, add:

```rust
#[test]
fn test_movegen_bb_matches_movegen_on_initial_position() {
    use crate::movegen;
    let board = Board::new(RuleSet::Official, 1);

    let old_moves = movegen::generate_pseudo_moves(&board, Color::Red);
    let new_moves = bitboards::generate_pseudo_moves_bitboards(&board, Color::Red);

    assert_eq!(old_moves.len(), new_moves.len());
    for action in &new_moves {
        assert!(old_moves.contains(action), "New movegen produced move not in old: {:?}", action);
    }
    for action in &old_moves {
        assert!(new_moves.contains(action), "Old movegen produced move not in new: {:?}", action);
    }
}
```

- [ ] **Step 2: Run test — expect it to FAIL (new function doesn't exist yet)**

Run: `cargo test test_movegen_bb_matches_movegen_on_initial_position -- --nocapture`
Expected: FAIL — function doesn't exist yet (but we're adding it in Task 11, so this test goes after Task 11)

Actually, add this test AFTER completing Task 11. The purpose is to validate the new bitboard movegen matches the old.

- [ ] **Step 3: Once Task 11+12 complete, run the test**

Run: `cargo test test_movegen_bb_matches_movegen_on_initial_position -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/bitboards.rs
git commit -m "test: add movegen equivalence test"
```

---

## Task 15: Migrate `eval_impl` functions to bitboard iteration

**Files:**
- Modify: `src/eval.rs` — migrate grid scans in game_phase, center_control, attack_rewards, piece_coordination, pawn_structure, elephant_structure, king_safety, handcrafted_evaluate

This is the largest task. Break it into sub-steps:

### 15a: Migrate `game_phase`

Read `src/eval.rs:189-200` (game_phase function).

Replace:
```rust
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
```

With:
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

### 15b: Migrate `center_control`

Replace the complex geometric formula with bitboard attack counts + `CORE_AREA_MASK`.

### 15c: Migrate remaining functions

Proceed similarly for each function — replace grid scans with `occupied(color)` + `lsb_index` loops, use bitboard attack methods.

- [ ] **Step: Run tests after each sub-migration**

Run: `cargo test` after each function migration
Expected: All PASS, eval scores unchanged

- [ ] **Step: Commit after each function**

---

## Self-Review Checklist

After writing the complete plan, run through:

1. **Spec coverage**: Every section in the spec has a corresponding task
2. **Placeholder scan**: No "TBD", "TODO", "implement later" anywhere
3. **Type consistency**: `horse_attacks(sq, color)` called consistently with color param; `Action` type matches across tasks

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-04-13-bitboard-movegen-eval-plan.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**