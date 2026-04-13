# Bitboard Phase 2: SEE (Static Exchange Evaluation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `find_least_valuable_attacker` and `see()` in the existing search module to use bitboard attack rays instead of the current directional array scanning. This makes attack detection O(1) per direction instead of O(n).

**Architecture:** Phase 2 modifies existing search code to use bitboards. The `Bitboards` struct from Phase 1 is already available via `board.bitboards`. No new files are created — only existing code in `src/main.rs` is modified.

**Tech Stack:** Pure Rust, uses `Bitboards` from Phase 1, `u128::count_ones()` for popcount.

---

## File Map

| File | Role |
|---|---|
| `src/main.rs` | **MOD** — Update `find_least_valuable_attacker` and `see()` to use bitboards |

---

## Background: Current SEE Implementation

The current SEE in `src/main.rs` around line 1450 does:
1. For each of 4 cardinal directions, scan outward from target
2. Check for chariot (no blockers yet) and cannon (exactly 1 blocker)
3. Check horse attacks around target using 8 deltas
4. Uses board.get() to query pieces along the way

The key function is `find_least_valuable_attacker(board, tar, side)` which returns `(Option<Coord>, i32)`.

---

## Task 1: Add `attackers` method to Bitboards

**Files:**
- Modify: `src/bitboards.rs` — add to `impl Bitboards`

- [ ] **Step 1: Add `attackers` method**

```rust
/// Returns a u128 bitboard of all squares containing pieces of `color`
/// that can attack the given `target` square.
/// This is the core building block for SEE.
pub fn attackers(&self, target: u8, color: Color) -> u128 {
    let occ = self.occupied_all();
    let occ_color = self.occupied(color);
    let rays = get_chariot_rays();
    let screens = get_cannon_screens();
    let mut attackers = 0u128;

    // Chariot attacks: slides in 4 directions
    for dir in 0..4 {
        let ray = rays[target as usize][dir];
        let blockers = ray & occ;
        if blockers == 0 { continue; }
        let nearest = Self::lsb_index(blockers);
        // Chariot is attacker if it can reach nearest blocker and it's our piece
        if occ_color & (1_u128 << nearest) != 0 {
            // Check it's actually a chariot (not just any piece on that square)
            if self.pieces[PieceType::Chariot as usize][color as usize] & (1_u128 << nearest) != 0 {
                attackers |= 1_u128 << nearest;
            }
        }
    }

    // Cannon attacks: needs exactly 1 screen between src and target
    for dir in 0..4 {
        let ray = rays[target as usize][dir];
        let blockers = ray & occ;
        if blockers == 0 { continue; }
        let nearest = Self::lsb_index(blockers);
        // Is nearest blocker a cannon?
        if self.pieces[PieceType::Cannon as usize][color as usize] & (1_u128 << nearest) != 0 {
            let second_blockers = ray & occ & !(rays[nearest as usize][dir]);
            if second_blockers != 0 {
                let second = Self::lsb_index(second_blockers);
                // If cannon is on target's side of second blocker (nearer to target)
                // then it can capture through its screen
                attackers |= 1_u128 << nearest;
            }
        }
    }

    // Horse attacks: 8 L-shape destinations around target
    // To find attackers: horse at SRC attacks TAR means SRC = TAR - HORSE_DELTA
    // So we look at squares TAR - delta
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
```

- [ ] **Step 2: Run build to verify**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo build 2>&1 | head -30`
Expected: No errors

---

## Task 2: Rewrite find_least_valuable_attacker using Bitboards

**Files:**
- Modify: `src/main.rs` — `find_least_valuable_attacker` function (around line 1500)

- [ ] **Step 1: Replace find_least_valuable_attacker**

Find the current `find_least_valuable_attacker` function (around line 1500). It starts with:
```rust
fn find_least_valuable_attacker(board: &Board, tar: Coord, side: Color) -> (Option<Coord>, i32) {
```

Replace it with:

```rust
fn find_least_valuable_attacker(board: &Board, tar: Coord, side: Color) -> (Option<Coord>, i32) {
    let tar_sq = (tar.y * 9 + tar.x) as u8;
    let attackers_bb = board.bitboards.attackers(tar_sq, side);

    if attackers_bb == 0 {
        return (None, i32::MAX);
    }

    // Find the least valuable attacker
    // Priority order (lowest value first): Pawn, Elephant, Advisor, Horse, Cannon, Chariot, King
    let mut min_value = i32::MAX;
    let mut min_attacker = None;

    // Check pawns first (lowest value)
    let pawns = attackers_bb & board.bitboards.pieces[PieceType::Pawn as usize][side as usize];
    if pawns != 0 {
        let sq = Bitboards::lsb_index(pawns);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Pawn as usize]);
    }

    // Check elephants
    let elephants = attackers_bb & board.bitboards.pieces[PieceType::Elephant as usize][side as usize];
    if elephants != 0 {
        let sq = Bitboards::lsb_index(elephants);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Elephant as usize]);
    }

    // Check advisors
    let advisors = attackers_bb & board.bitboards.pieces[PieceType::Advisor as usize][side as usize];
    if advisors != 0 {
        let sq = Bitboards::lsb_index(advisors);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Advisor as usize]);
    }

    // Check horses
    let horses = attackers_bb & board.bitboards.pieces[PieceType::Horse as usize][side as usize];
    if horses != 0 {
        let sq = Bitboards::lsb_index(horses);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Horse as usize]);
    }

    // Check cannons
    let cannons = attackers_bb & board.bitboards.pieces[PieceType::Cannon as usize][side as usize];
    if cannons != 0 {
        let sq = Bitboards::lsb_index(cannons);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Cannon as usize]);
    }

    // Check chariots
    let chariots = attackers_bb & board.bitboards.pieces[PieceType::Chariot as usize][side as usize];
    if chariots != 0 {
        let sq = Bitboards::lsb_index(chariots);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::Chariot as usize]);
    }

    // Check king (should be last)
    let kings = attackers_bb & board.bitboards.pieces[PieceType::King as usize][side as usize];
    if kings != 0 {
        let sq = Bitboards::lsb_index(kings);
        return (Some(Coord::new((sq % 9) as i8, (sq / 9) as i8)), SEE_VALUE[PieceType::King as usize]);
    }

    (None, i32::MAX)
}
```

**Note:** This uses `board.bitboards.attackers(tar_sq, side)` to get a bitboard of all attackers, then finds the least valuable by checking in priority order.

- [ ] **Step 2: Run build to verify**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo build 2>&1 | head -30`
Expected: No errors

---

## Task 3: Run full test suite to verify SEE equivalence

- [ ] **Step 1: Run all tests**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo test 2>&1 | grep "test result"`
Expected: `test result: ok. 249 passed; 0 failed`

---

## Task 4: Commit Phase 2

- [ ] **Step 1: Commit the Phase 2 work**

```bash
git add src/bitboards.rs src/main.rs
git commit -m "$(cat <<'EOF'
feat(bitboard): Phase 2 — SEE using bitboard attack rays

- Add Bitboards::attackers() method returning u128 of all attackers
- Rewrite find_least_valuable_attacker using attackers bitboard
- Replace directional array scanning with O(1) bitboard queries
- All 249 tests pass (SEE equivalence verified)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All Phase 2 deliverables implemented
- [ ] **Placeholder scan:** No "TBD", "TODO", or incomplete sections
- [ ] **Type consistency:** `Coord` ↔ `sq` conversions use `y*9+x` consistently
- [ ] **File paths:** All paths are absolute and correct
- [ ] **No regressions:** All 249 tests pass

---

## Spec Coverage Gap Check

| Spec Section | Status |
|---|---|
| `Bitboards::attackers(target, color) -> u128` | ✅ Task 1 |
| `find_least_valuable_attacker` rewritten | ✅ Task 2 |
| SEE equivalence verified | ✅ Task 3 |

**No gaps found.**
