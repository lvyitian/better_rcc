# OpeningBook & EndgameTablebase Fixes — Design Spec

## Status

Approved 2026-04-10. Implementation pending.

---

## 1. OpeningBook Root Position Fix

### Problem

Seven `init_*` functions each write their first move to the starting position's Zobrist hash via `self.book.insert(key, a1)`. Since all openings share the identical starting position, each initializer overwrites the previous one's entry. Only the last initializer's first move survives.

### Solution

Refactor `init_all_openings()` to:

1. Build the starting board once
2. Collect all unique first moves **before** any board mutations
3. Insert them as a deduplicated alternatives list at the root position
4. Each continuation function (`init_*_continuation()`) builds only from ply 2+ (ply 1 is already recorded)

**Deduplication**: Some openings share the same first move (e.g., 炮二平五 appears in both dang tou pao and shun pao; 马八进七 appears in lie pao and qi ma). The merged list uses a `BTreeSet`-like dedup by inserting into a `Vec` only if the move is not already present.

### Implementation

```rust
fn init_all_openings(&mut self) {
    let mut board = Board::new(RuleSet::Official, 1);
    let root_key = board.zobrist_key;

    // Collect unique first moves
    let mut root_moves = Vec::new();
    let add_if_new = |moves: &mut Vec<_>, m: Action| {
        if !moves.contains(&m) { moves.push(m); }
    };

    add_if_new(&mut root_moves, Action::new(Coord::new(7, 7), Coord::new(4, 7), None)); // 炮二平五
    add_if_new(&mut root_moves, Action::new(Coord::new(1, 9), Coord::new(2, 7), None)); // 马八进七
    add_if_new(&mut root_moves, Action::new(Coord::new(6, 9), Coord::new(4, 7), None)); // 相三进五
    add_if_new(&mut root_moves, Action::new(Coord::new(1, 7), Coord::new(3, 7), None)); // 炮八平七
    add_if_new(&mut root_moves, Action::new(Coord::new(6, 6), Coord::new(6, 5), None)); // 兵三进一

    self.insert(root_key, &root_moves);

    // Build each line's continuation from ply 2
    self.build_dang_tou_pao_continuation(&mut board);
    self.build_shun_pao_continuation(&mut board);
    self.build_lie_pao_continuation(&mut board);
    self.build_fei_xiang_continuation(&mut board);
    self.build_qi_ma_continuation(&mut board);
    self.build_guo_gong_pao_continuation(&mut board);
    self.build_xian_ren_zhi_lu_continuation(&mut board);
}
```

Each `build_*_continuation()` function replays the first move internally (if needed for that line's chain) and records subsequent moves. All `self.book.insert` calls in initializers are replaced with `self.insert` for consistency.

---

## 2. EndgameTablebase Phase Blending

### Problem

`EndgameTablebase::probe()` returns a raw centipawn score that short-circuits the entire evaluation function. A 17-point chariot advantage is reported identically whether 4 or 30 pieces remain, overstating certainty in complex positions.

### Solution

1. Change `probe` signature to return `Option<(i32, f32)>` — raw score and a confidence weight.
2. Compute confidence as `1.0 - (total_pieces as f32 / 32.0)`, clamped to `[0.0, 1.0]`. Fewer pieces → higher confidence.
3. In `evaluate()`, blend the tablebase score with the regular evaluation based on confidence:
   - If `confidence >= 0.5`: mostly tablebase, lightly blended with eval
   - If `confidence < 0.5`: mostly regular eval, lightly influenced by tablebase
   - If `confidence < 0.2`: treat as fallback only (return tablebase score directly)

```rust
// In eval::evaluate():
if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
    let fallback = /* compute regular eval */;
    if conf < 0.2 {
        return tb_score;
    }
    // Linear interpolation: weight = conf
    let blended = (tb_score as f32 * conf + fallback as f32 * (1.0 - conf)) as i32;
    return blended;
}
```

**Confidence formula**: `conf = max(0.0, min(1.0, 1.0 - total_pieces / 32.0))`
- 32 pieces: conf = 0.0 (don't trust simplified tablebase in opening/middlegame)
- 16 pieces: conf = 0.5
- 4 pieces: conf = 0.875

---

## 3. New Endgame Tablebase Patterns

Add 5 new patterns to `EndgameTablebase`. Scores are in centipawns from Red's perspective.

### 3.1 Chariot vs Chariot (Drawish)

```rust
fn check_chariot_vs_chariot(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
    if red[PieceType::King as usize] == 1
        && red[PieceType::Chariot as usize] == 1
        && red_other == 1
        && black[PieceType::King as usize] == 1
        && black[PieceType::Chariot as usize] == 1
        && black_other == 1
    {
        // Symmetrical. Slight edge for side to move (initiative).
        // Return small non-zero score based on side to move.
        let score = 5000; // ~half a pawn
        return Some(if side == Color::Red { score } else { -score });
    }
    None
}
```

### 3.2 Rook + 2 Advisors vs Rook (Red winning)

```rust
fn check_rook_2advisors_vs_rook(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
    if red[PieceType::King as usize] == 1
        && red[PieceType::Chariot as usize] == 1  // Rook = Chariot in Xiangqi
        && red[PieceType::Advisor as usize] == 2
        && red_other == 3  // 1 chariot + 2 advisors = 3 "other"
        && black[PieceType::King as usize] == 1
        && black[PieceType::Chariot as usize] == 1
        && black_other == 1
    {
        let score = 60000;
        return Some(if side == Color::Red { score } else { -score });
    }
    None
}
```

### 3.3 Rook vs 2 Advisors (Defensive drawish)

```rust
fn check_rook_vs_2advisors(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
    if red[PieceType::King as usize] == 1
        && red[PieceType::Chariot as usize] == 1
        && red_other == 1
        && black[PieceType::King as usize] == 1
        && black[PieceType::Advisor as usize] == 2
        && black_other == 2
    {
        // Defensive material equality, but advisors are passive.
        let score = -40000;
        return Some(if side == Color::Red { score } else { -score });
    }
    None
}
```

### 3.4 Two Elephants vs Nothing

```rust
fn check_two_elephants_vs_nothing(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
    if red[PieceType::King as usize] == 1
        && red[PieceType::Elephant as usize] == 2
        && red_other == 2
        && black[PieceType::King as usize] == 1
        && black_other == 0
    {
        // Solid defensive material, but no attack. Small edge.
        let score = 35000;
        return Some(if side == Color::Red { score } else { -score });
    }
    None
}
```

### 3.5 Horse + Pawn vs Advisor

```rust
fn check_horse_pawn_vs_advisor(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
    if red[PieceType::King as usize] == 1
        && red[PieceType::Horse as usize] == 1
        && red[PieceType::Pawn as usize] == 1
        && red_other == 2
        && black[PieceType::King as usize] == 1
        && black[PieceType::Advisor as usize] == 1
        && black_other == 1
    {
        // Horse+pawn is slightly stronger than advisor alone in simplified EG.
        let score = 55000;
        return Some(if side == Color::Red { score } else { -score });
    }
    None
}
```

### 3.6 Probe Order

New patterns added to `probe()` after the existing 7 checks:
1. `check_double_chariot_vs_single` (existing)
2. `check_chariot_cannon_vs_chariot` (existing)
3. `check_pawn_vs_advisor` (existing)
4. `check_horse_cannon_vs_double_advisor` (existing)
5. `check_horse_vs_advisor` (existing)
6. `check_cannon_advisor_vs_advisor` (existing)
7. `check_chariot_vs_defense` (existing)
8. `check_chariot_vs_chariot` (new)
9. `check_rook_2advisors_vs_rook` (new)
10. `check_rook_vs_2advisors` (new)
11. `check_two_elephants_vs_nothing` (new)
12. `check_horse_pawn_vs_advisor` (new)

---

## 4. OpeningBook probe() Legality Fix

### Problem

The `probe()` legality check only verifies occupancy (piece at src, correct color at dst). It does not verify the move doesn't leave the king in check.

### Solution

After filtering candidates to legal occupancy moves, perform an `is_legal_move` check before returning:

```rust
let valid_moves: Vec<Action> = candidates.into_iter()
    .filter(|a| {
        board.get(a.src).is_some()
            && (board.get(a.tar).is_none()
                || board.get(a.tar).is_some_and(|p| p.color != board.current_side))
    })
    .filter(|a| {
        let (legal, _) = movegen::is_legal_move(board, *a, board.current_side);
        legal
    })
    .collect();
```

This adds a `is_legal_move` call per candidate. Since `probe` only runs at the root and the number of candidates is small (1 primary + a few alternatives), the performance cost is negligible.

---

## 5. Consistent `insert` Usage

### Problem

Some initializer functions use `self.book.insert(key, action)` directly while others use `self.insert(key, &[action])`. The former bypasses deduplication logic.

### Solution

Replace all `self.book.insert` calls in all initializer functions with `self.insert(key, &[action])`.

---

## Files Affected

- `src/main.rs` — `book` module (OpeningBook struct, init functions, probe, insert)
- `src/main.rs` — `eval::evaluate()` function (blending logic)
- `src/main.rs` — `EndgameTablebase` impl block (new patterns, probe signature change)

---

## Testing

- Opening book: verify starting position has all 5 unique first moves as alternatives
- Opening book: verify `probe` returns a legal move for each first-move option
- Endgame tablebase: verify `probe` returns `(score, confidence)` tuple
- Endgame tablebase: verify blending produces intermediate scores near phase boundaries
- All 5 new patterns: create simplified board matching each pattern, verify `probe` returns correct score
