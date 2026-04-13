# Bitboard Phase 4: Remove cells Array — Bitboards as Canonical State

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `cells: [[Option<Piece>; 9]; 10]` from `Board`, making `Bitboards` the canonical state. Replace `king_pos: RefCell<[Option<Coord>; 2]>` cache with `Bitboards::king_pos()` using bitscan. The `move_history` stays as `Vec<Action>` — only use `SmallVec` in bitboards.rs (already in use there).

**Architecture:** Phase 4 is the final payoff — the bitboards representation becomes primary and `cells` is removed entirely. This eliminates ~3600 bytes of redundant board state per `Board` instance and simplifies the codebase.

**Tech Stack:** Pure Rust. Uses `Bitboards::lsb_index()` for O(1) king position lookup.

---

## File Map

| File | Role |
|---|---|
| `src/bitboards.rs` | **MOD** — Add `king_pos`, `find_kings`, `piece_at` methods |
| `src/main.rs` | **MOD** — Remove `cells` and `king_pos` from `Board`, update all usages |

---

## Background: Current State

After Phase 3, `Board` has both:
- `cells: [[Option<Piece>; 9]; 10]` — canonical (used by all existing code)
- `bitboards: Bitboards` — shadow (kept in sync, used for validation)

`king_pos: RefCell<[Option<Coord>; 2]>` caches king positions for O(1) lookup.

Phase 4 makes `Bitboards` the only state and removes `cells` entirely. `king_pos` cache is replaced by `Bitboards::king_pos(color)` which uses `lsb_index` on the king bitboard.

---

## Task 1: Change Vec to SmallVec in bitboards.rs, add king_pos/find_kings/piece_at

**Files:**
- Modify: `src/bitboards.rs` — change Vec→SmallVec, add new methods

- [ ] **Step 1: Add SmallVec import and change Vec usages**

In `bitboards.rs`, add at the top (near the existing imports):
```rust
use smallvec::SmallVec;
```

Change `generate_moves` return type and construction:
```rust
// Change return type from Vec<u8> to SmallVec<[u8; 17]>
pub fn generate_moves(&self, from: u8, color: Color) -> SmallVec<[u8; 17]> {
    let mut destinations = SmallVec::new();  // capacity 17 fits in inline storage
```

Change `generate_pseudo_moves` return type and construction:
```rust
// Change return type from Vec<(u8, u8, Option<Piece>)> to SmallVec<[(u8, u8, Option<Piece>); 64]>
pub fn generate_pseudo_moves(&self, color: Color) -> SmallVec<[(u8, u8, Option<Piece>); 64]> {
    let mut moves = SmallVec::new();  // capacity 64 fits in inline storage
```

Note: `SmallVec<[T; N]>` stores up to N elements inline without heap allocation. For `generate_moves` max is 17 (chariot lines), for `generate_pseudo_moves` max is 64 (16 pieces × 4 max destinations average). Both comfortably fit inline.

- [ ] **Step 2: Add the king_pos, find_kings, and piece_at methods**

Add these methods before the tests module at the end of `bitboards.rs`:

```rust
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

/// Returns the piece at the given square index (0-89), or None if empty.
/// This is O(1) — just checks 14 bitboard bits.
#[inline(always)]
pub fn piece_at(&self, sq: u8) -> Option<Piece> {
    let bit = 1_u128 << sq;
    // Check red pieces first
    let occ_red = self.occupied(Color::Red);
    if occ_red & bit != 0 {
        for pt in 0..7 {
            if self.pieces[pt][Color::Red as usize] & bit != 0 {
                return Some(Piece {
                    color: Color::Red,
                    piece_type: unsafe { std::mem::transmute(pt as u8) },
                });
            }
        }
    }
    // Check black pieces
    let occ_black = self.occupied(Color::Black);
    if occ_black & bit != 0 {
        for pt in 0..7 {
            if self.pieces[pt][Color::Black as usize] & bit != 0 {
                return Some(Piece {
                    color: Color::Black,
                    piece_type: unsafe { std::mem::transmute(pt as u8) },
                });
            }
        }
    }
    None
}
```

Note: We could use `PieceType::from_repr(pt)` but since we're iterating 0..7 and all those values are valid `PieceType` variants, `transmute` is safe here and avoids the `from_repr` Option unwrap overhead. If `from_repr` is available and panic-safe, prefer that.

Actually, check if `PieceType` has a `from_repr` or similar — if not, use:
```rust
piece_type: [PieceType::King, PieceType::Advisor, PieceType::Elephant, PieceType::Horse, PieceType::Cannon, PieceType::Chariot, PieceType::Pawn][pt]
```
This is safe because pt is in 0..7.

- [ ] **Step 2: Run build to verify**

Run: `cargo build 2>&1 | head -30`
Expected: No errors

---

## Task 2: Remove cells and king_pos from Board struct and update all usages

**Files:**
- Modify: `src/main.rs`

**Step 1: Remove `cells` field from Board struct**

Remove from `Board` struct definition (around line 1588):
```rust
pub cells: [[Option<Piece>; 9]; 10],  // cells[y][x], None = empty
```

**Step 2: Remove `king_pos` field from Board struct**

Remove from `Board` struct definition (around line 1597):
```rust
pub king_pos: RefCell<[Option<Coord>; 2]>,
```
Also remove the comment above it:
```rust
// Cached king positions [Red, Black] for O(1) lookup instead of O(90) scan
// None = cache invalid, Some(pos) = position known
```

**Step 3: Remove `king_pos: RefCell::new([None, None])` from Board::new()**

Remove from `Board::new()` (around line 1677).

**Step 4: Update `set_internal` method**

The `set_internal` method currently updates `cells` and the king cache. Since `cells` is gone, replace the body. The method is used by SEE to make temporary moves. Replace `set_internal` with:

```rust
/// Internal helper to set a piece and update Zobrist hash only.
/// Since cells is removed, this only updates the Zobrist hash.
/// Bitboards are NOT updated here — callers must use bitboards.apply_move.
#[inline(always)]
fn set_internal(&mut self, coord: Coord, piece: Option<Piece>) {
    let sq = coord.y as usize * 9 + coord.x as usize;
    let zobrist = get_zobrist();

    // Remove old piece from hash
    if let Some(old) = self.get(coord) {
        let old_idx = zobrist.pos_idx(coord);
        self.zobrist_key ^= zobrist.pieces[old_idx][old.color as usize][old.piece_type as usize];
    }

    // Add new piece to hash
    if let Some(new_piece) = piece {
        let new_idx = zobrist.pos_idx(coord);
        self.zobrist_key ^= zobrist.pieces[new_idx][new_piece.color as usize][new_piece.piece_type as usize];
    }

    // Note: We do NOT update bitboards here.
    // The bitboards are updated by the caller via bitboards.apply_move(...)
    // This is only used for Zobrist hash updates in SEE.
}
```

Wait — if we don't update `cells` but we need `self.get(coord)` to work, then `get()` must use `bitboards.piece_at()`. That's fine because we're updating `get()` next.

Actually, `set_internal` is called in the SEE loop. Let me re-read the SEE code to understand the flow.

In SEE (around line 2100+):
```rust
board_copy.set_internal(dst, None);  // remove captured piece from hash
board_copy.bitboards.apply_move(Action { ... });  // sync bitboards
```

After Phase 4, `set_internal` only updates the Zobrist hash (no cells), and the bitboards are updated by `apply_move`. The `set_internal` no longer needs to clear the piece from cells because there are no cells.

**Step 5: Update `Board::get()` to use bitboards**

Replace the body (around line 1798):
```rust
#[inline(always)]
pub fn get(&self, coord: Coord) -> Option<Piece> {
    if coord.is_valid() {
        self.bitboards.piece_at(coord.y as u8 * 9 + coord.x as u8)
    } else {
        None
    }
}
```

**Step 6: Update `find_kings()` method**

Replace the entire `find_kings` method body (around line 1956) with:
```rust
#[inline(always)]
pub fn find_kings(&self) -> (Option<Coord>, Option<Coord>) {
    self.bitboards.find_kings()
}
```

**Step 7: Remove `invalidate_king_cache()` method**

Remove this method entirely (around lines 2012-2015) since the cache no longer exists.

**Step 8: Update `from_fen`**

In `from_fen()`:
- Keep the local `cells` temporary array (it's used to build bitboards via `Bitboards::from_cells(&cells)`)
- Remove the king_pos cache initialization (lines ~1769-1784) — kings are found via bitscan
- Remove `cells` from the Board construction (line ~1787) — don't include it in the struct
- Change the `king_pos` field to be omitted from the struct initialization

The `from_fen` should build a local `cells` array, then:
```rust
Board {
    bitboards: Bitboards::from_cells(&cells),
    zobrist_key,
    current_side,
    rule_set: RuleSet::Official,
    move_history: Vec::with_capacity(200),
    repetition_history,
    // NO king_pos field
}
```

**Step 9: Update `make_board` test helper**

In the test helper `make_board()` (around line 3969), remove `cells` and `king_pos` fields from the Board construction. Since `make_board` creates a Board with specific pieces, it should use `Bitboards::from_cells(&cells)` just like `from_fen` does.

Change:
```rust
Board {
    cells,
    bitboards: Bitboards::from_cells(&cells),
    zobrist_key,
    current_side,
    rule_set: RuleSet::Official,
    move_history: vec![],
    repetition_history: HashMap::new(),
    king_pos: RefCell::new([None, None]),
}
```
to:
```rust
Board {
    bitboards: Bitboards::from_cells(&cells),
    zobrist_key,
    current_side,
    rule_set: RuleSet::Official,
    move_history: vec![],
    repetition_history: HashMap::new(),
}
```

**Step 10: Update all other `self.cells` usages**

Search for `self.cells` in main.rs and replace each:

1. `set_internal` — already handled in Step 4
2. `find_kings` — already handled in Step 6
3. `make_move` — check if any `self.cells` usages remain
4. Any other `cells` usages — replace with bitboard equivalents

Use `git diff` or search to find all remaining `self.cells` references and replace them with `self.bitboards.piece_at(...)` or other appropriate bitboard calls.

**Step 11: Verify no cells references remain**

Run: `grep -n "self\.cells" src/main.rs` or equivalent
Expected: No matches

---

## Task 3: Run tests

- [ ] **Step 1: Run tests**

Run: `cargo test 2>&1 | grep "test result"`
Expected: All tests pass

If there are failures, debug and fix each one. Common issues:
- `piece_at` not implemented correctly
- `set_internal` Zobrist hash not matching expected
- `from_fen` not building bitboards correctly
- `make_board` test helper missing fields

---

## Task 4: Commit Phase 4

- [ ] **Step 1: Commit the Phase 4 work**

```bash
git add src/bitboards.rs src/main.rs
git commit -m "$(cat <<'EOF'
feat(bitboard): Phase 4 — Remove cells array, Bitboards is canonical

- Add Bitboards::king_pos(color) and find_kings() using bitscan
- Add Bitboards::piece_at(sq) for O(1) piece lookup
- Remove cells: [[Option<Piece>; 9]; 10] from Board struct
- Remove king_pos: RefCell cache — kings found via bitscan
- Board::get() now uses bitboards.piece_at()
- All tests pass

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All Phase 4 deliverables implemented
- [ ] **Placeholder scan:** No "TBD", "TODO", or incomplete sections
- [ ] **cells removed:** No `self.cells` references remain in main.rs
- [ ] **king_pos cache removed:** No `RefCell<[Option<Coord>; 2]>` in Board struct
- [ ] **No regressions:** All tests pass

---

## Spec Coverage Gap Check

| Spec Section | Status |
|---|---|
| Remove `cells` array from Board | ✅ Task 2 |
| `Bitboards::king_pos(color)` using bitscan | ✅ Task 1 |
| `Bitboards::find_kings()` | ✅ Task 1 |
| `Bitboards::piece_at(sq)` | ✅ Task 1 |
| Remove `king_pos` RefCell cache | ✅ Task 2 |
| Board::get() uses bitboards | ✅ Task 2 |
| All existing tests pass | ✅ Task 3 |

**No gaps found.**
