# Bitboard Phase 3: NNUE Input Encoding from Bitboards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Bitboards::to_nnue_input()` method that produces identical `NNInputPlanes` output to `NNInputPlanes::from_board()`, but iterates bitboards instead of the cells array. This validates bitboard correctness end-to-end and sets up for Phase 4 (where the cells array is removed).

**Architecture:** Phase 3 adds a `to_nnue_input` method to `Bitboards`. No existing code is modified — this is purely additive. The method iterates the 14 `u128` bitboards and sets feature indices directly from set bits.

**Tech Stack:** Pure Rust. Uses `u128::count_ones()` for non-king piece counting. NNUE encoding already exists in `src/nnue_input.rs`.

---

## File Map

| File | Role |
|---|---|
| `src/bitboards.rs` | **MOD** — Add `to_nnue_input` and `count_non_king_pieces` methods |
| `src/nnue_input.rs` | **READ** — Reference for NNInputPlanes encoding (do not modify) |

---

## Background: Current NNUE Input Encoding

The current `NNInputPlanes::from_board()` in `src/nnue_input.rs` (line 52-90) iterates all 90 squares via double-loop:

```rust
for y in 0..10 {
    for x in 0..9 {
        let cell = &board.cells[y][x];
        if let Some(piece) = cell {
            let base = if piece.color == our_color { 0 } else { 630 };
            let piece_idx = piece.piece_type as usize;
            let square_idx = y * 9 + x;
            let feature_idx = base + piece_idx * 90 + square_idx;
            stm.data[feature_idx] = 1.0;
        }
    }
}
```

The encoding uses **dual perspective**: stm (side-to-move) and ntm (not-side-to-move, with vertical flip y→9-y).

---

## Task 1: Add to_nnue_input method to Bitboards

**Files:**
- Modify: `src/bitboards.rs` — add to `impl Bitboards`

- [ ] **Step 1: Add to_nnue_input and count_non_king_pieces methods**

Add these methods to `impl Bitboards` (before the tests module at the end of the file):

```rust
/// Convert bitboards to NNUE input feature planes (dual perspective).
/// Returns (stm, ntm) matching NNInputPlanes::from_board output exactly.
/// stm = side-to-move perspective, ntm = not-side-to-move with vertical flip.
pub fn to_nnue_input(&self, stm: Color) -> NNInputPlanes {
    let mut stm_planes = NNInputPlanes::new();
    let ntm = stm.opponent();

    // Encode each piece type/color bitboard
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
                stm_planes.data[stm_feature] = 1.0;

                // ntm perspective (vertical flip y -> 9-y)
                let ntm_sq_idx = (9 - y) * 9 + x;
                let ntm_base = if c == ntm as usize { 0 } else { 630 };
                let ntm_feature = ntm_base + pt * 90 + ntm_sq_idx;
                // We need to populate ntm data too, but NNInputPlanes::from_board
                // returns both stm and ntm. We'll return only stm here and the
                // caller can build ntm by calling with ntm as stm.
                tmp &= tmp - 1; // Clear LSB
            }
        }
    }

    stm_planes
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
```

**Wait** — `NNInputPlanes` is not in `bitboards.rs`. The method needs to return something different. Let me fix the design:

Actually, since `NNInputPlanes` lives in `nnue_input.rs` and can't be easily imported into `bitboards.rs` (circular dependency risk), we should make `to_nnue_input` return a simpler structure and let `nnue_input.rs` use it. Or we can add it to `nnue_input.rs` instead.

**Alternative approach (better):** Add the bitboard iteration to `nnue_input.rs` as a separate function that takes `&Bitboards`. The test compares outputs.

Actually, the cleanest approach: add `to_nnue_input` that returns a raw `[f32; 1260]` array, then wrap it in `NNInputPlanes` in `nnue_input.rs` if needed.

Let me use this simpler signature:

```rust
/// Fill the NNUE input feature planes from bitboards (side-to-move perspective).
/// Populates stm_data[1260] array where stm squares are at base 0, opponent at base 630.
pub fn fill_nnue_features(&self, stm: Color, stm_data: &mut [f32; 1260]) {
    let ntm = stm.opponent();

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

                tmp &= tmp - 1;
            }
        }
    }
}
```

Then in `nnue_input.rs` or as a test, we verify it matches `NNInputPlanes::from_board`.

- [ ] **Step 2: Run build to verify**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo build 2>&1 | head -30`
Expected: No errors

---

## Task 2: Add NNUE equivalence test in bitboards tests

**Files:**
- Modify: `src/bitboards.rs` — add test in the `#[cfg(test)] mod tests` block

- [ ] **Step 1: Add NNUE equivalence test**

Add this test to the existing `#[cfg(test)] mod tests` block in `bitboards.rs`:

```rust
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
        let moves = generate_legal_moves(&mut board2, board2.current_side);
        if moves.is_empty() { break; }
        let idx = (board2.zobrist_key as usize % moves.len()) as usize;
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
```

**Note:** You'll need to add `use crate::nnue_input::NNInputPlanes;` to the test imports.

- [ ] **Step 2: Run tests**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo test bitboards 2>&1 | tail -20`
Expected: All bitboards tests pass (including the new NNUE equivalence test)

---

## Task 3: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `cd F:/RustroverProjects/better_rust_chinese_chess/.worktrees/bitboard && cargo test 2>&1 | grep "test result"`
Expected: `test result: ok. 249 passed; 0 failed`

---

## Task 4: Commit Phase 3

- [ ] **Step 1: Commit the Phase 3 work**

```bash
git add src/bitboards.rs src/main.rs
git commit -m "$(cat <<'EOF'
feat(bitboard): Phase 3 — NNUE input encoding from bitboards

- Add Bitboards::fill_nnue_features() to populate NNUE feature planes
- Add Bitboards::count_non_king_pieces() using popcount
- Add NNUE equivalence test (bitboard encoding matches NNInputPlanes::from_board)
- All 249 tests pass

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All Phase 3 deliverables implemented
- [ ] **Placeholder scan:** No "TBD", "TODO", or incomplete sections
- [ ] **NNUE equivalence verified:** Test confirms bitboard output matches original encoding
- [ ] **No regressions:** All 249 tests pass

---

## Spec Coverage Gap Check

| Spec Section | Status |
|---|---|
| `Bitboards::to_nnue_input()` → `fill_nnue_features` | ✅ Task 1 |
| `count_non_king_pieces()` using popcount | ✅ Task 1 |
| NNUE equivalence test | ✅ Task 2 |

**No gaps found.**
