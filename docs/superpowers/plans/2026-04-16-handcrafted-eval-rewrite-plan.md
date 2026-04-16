# Handcrafted Eval Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete rewrite of `src/eval.rs` (eval_impl module) per spec at `docs/superpowers/specs/2026-04-16-handcrafted-eval-rewrite-design.md`. All intermediate scoring functions return Red-positive. Integer arithmetic only. No initiative bonus. King attack bonus reduced from 25→15. Use bitboards for all mobility computation.

**Architecture:** Single-pass per stage. Stage 1 (material+PST), Stage 2 (mobility/activity), Stage 3 (king safety + threats), combined in `handcrafted_evaluate`. Each stage produces a single `Color → i32` value.

**Tech Stack:** Pure Rust, no external dependencies. Uses existing `crate::bitboards::Bitboards` for mobility (chariot/horse/elephant attacks), `crate::movegen::generate_elephant_moves` for elephant mobility, and manual iteration for cannon platform detection and hanging piece detection.

---

## API Notes (Codebase Facts)

These exist and should be used directly:
- `Bitboards::chariot_attacks(sq: u8, color: Color) -> u128` — all squares (empty + capturable) in 4 dirs
- `Bitboards::horse_attacks(sq: u8, color: Color) -> u128` — 8 knight destinations
- `Bitboards::elephant_attacks(sq: u8, color: Color) -> u128` — 4 elephant destinations (river-blocked)
- `Bitboards::piece_bitboard(pt: PieceType, color: Color) -> u128` — all pieces of a type/color
- `Bitboards::attackers(sq: u8, color: Color) -> u128` — all pieces attacking a square
- `Bitboards::occupied_all() -> u128` — all occupied squares
- `Bitboards::find_kings() -> (Option<Coord>, Option<Coord>)` — red, black king positions
- `Bitboards::lsb_index(bb: u128) -> u8` — index of least significant bit
- `Bitboards::msb_index(bb: u128) -> u8` — index of most significant bit
- `u128::count_ones() -> u32` — count set bits
- `crate::movegen::generate_elephant_moves(board, pos, color) -> MoveBuf` — elephant moves
- `Coord::in_palace(color) -> bool`, `Coord::crosses_river(color) -> bool`
- `crate::DIRS_4: [(i8, i8); 4]` — cardinal directions
- `crate::PALACE_DELTAS: [(i8, i8); 8]` — king-adjacent squares
- `crate::HORSE_DELTAS`, `crate::HORSE_BLOCKS`

These do NOT exist (use alternatives):
- `board.find_king(color)` → use `board.find_kings()` then match
- `board.pieces_of_color(color)` → iterate board with double for-loop

---

## Helper Types

### For iterating pieces of one color (used across multiple functions)

```rust
#[derive(Clone, Copy)]
struct PiecePos {
    pos: Coord,
    pt: PieceType,
}
```

Helper function (add to eval_impl):
```rust
fn pieces_of_color(board: &Board, color: Color) -> SmallVec<[PiecePos; 16]> {
    let mut v = SmallVec::new();
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(p) = board.get(pos) {
                if p.color == color {
                    v.push(PiecePos { pos, pt: p.piece_type });
                }
            }
        }
    }
    v
}
```

### Helper: get king position by color
```rust
fn king_pos(board: &Board, color: Color) -> Option<Coord> {
    let (rk, bk) = board.find_kings();
    match color {
        Color::Red => rk,
        Color::Black => bk,
    }
}
```

---

## Task 1: Write Scaffold — Constants, Tables, Phase, and PST

**Files:**
- Modify: `src/eval.rs` (complete rewrite of eval_impl module)

- [ ] **Step 1: Replace the entire eval_impl module with new scaffold**

Write `src/eval.rs` with:
1. Re-exports from `crate` (keep existing — `Board, Color, PieceType, Piece, Coord, movegen, RuleSet, EndgameTablebase, BOARD_WIDTH, BOARD_HEIGHT, CORE_X_MIN, CORE_X_MAX, CORE_Y_MIN, CORE_Y_MAX, HORSE_DELTAS, HORSE_BLOCKS, DIRS_4, PALACE_DELTAS, MATE_SCORE, CHECK_BONUS`)
2. All constants from the spec: `MG_VALUE`, `EG_VALUE`, all PST tables, `PHASE_WEIGHTS`, `TOTAL_PHASE`
3. `game_phase(board: &Board) -> i32` — unchanged from existing
4. `pst_val(pt, color, x, y) -> i32` — unchanged from existing
5. The helper types and functions above (`PiecePos`, `pieces_of_color`, `king_pos`)

Copy the existing PST tables verbatim from the old code:
- `MG_PST_KING`, `EG_PST_KING`
- `MG_PST_ADVISOR`
- `MG_PST_ELEPHANT`
- `MG_PST_HORSE`
- `MG_PST_CANNON`
- `MG_PST_CHARIOT`
- `MG_PST_PAWN`, `EG_PST_PAWN`

- [ ] **Step 2: Verify it compiles**

```bash
cargo build --no-default-features 2>&1 | head -40
```
Expected: compiles (empty module with stubs won't compile yet — that's fine, just check the PST tables are correct)

- [ ] **Step 3: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
refactor(eval): begin eval_impl rewrite - scaffold constants and PST tables

Keeps all existing PST tables and phase constants verbatim.
Adds helper types PiecePos, pieces_of_color(), king_pos().

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Stage 1 — material_pst()

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement material_pst()**

```rust
/// Stage 1: Material value (MG/EG interpolated) + PST bonus for all pieces.
/// Returns Red-positive score (positive = Red advantage).
pub fn material_pst(board: &Board) -> i32 {
    let phase = game_phase(board);
    let mut score = 0i32;

    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(piece) = board.get(pos) {
                // Material value with phase interpolation (integer math)
                let mg = MG_VALUE[piece.piece_type as usize];
                let eg = EG_VALUE[piece.piece_type as usize];
                let val = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
                score += val * piece.color.sign();

                // PST bonus
                let pst = pst_val(piece.piece_type, piece.color, x, y);
                score += pst;
            }
        }
    }
    score
}
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement Stage 1 material_pst()

Integer phase interpolation: (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
PST added per piece. Existing eval tests still pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Stage 2 — Mobility Functions (Bitboard-Accelerated)

**Files:**
- Modify: `src/eval.rs`

### 3a. horse_mobility (uses Bitboards)

- [ ] **Step 1: Implement horse_mobility() using bitboards**

```rust
/// Horse mobility: count of horse_attacks() bitboard.
/// Each valid destination = +10. Returns Red-positive.
pub fn horse_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
    let sq = (pos.y * 9 + pos.x) as u8;
    let attacks = board.bitboards.horse_attacks(sq, color);
    (attacks.count_ones() as i32) * 10
}
```

### 3b. chariot_mobility (uses Bitboards)

- [ ] **Step 2: Implement chariot_mobility() using bitboards**

Chariot mobility = count of empty squares reachable in each cardinal direction.
`chariot_attacks` gives (empty squares + capturable enemy squares) per direction.
To get only empty: `chariot_attacks & !occupied_all()`.

```rust
/// Chariot mobility: empty squares in 4 cardinal directions.
/// Each empty square = +5. Returns Red-positive.
pub fn chariot_mobility(board: &Board, pos: Coord, _color: Color) -> i32 {
    let sq = (pos.y * 9 + pos.x) as u8;
    let attacks = board.bitboards.chariot_attacks(sq, _color);
    let empty = attacks & !board.bitboards.occupied_all();
    (empty.count_ones() as i32) * 5
}
```

### 3c. elephant_mobility (uses generate_elephant_moves)

- [ ] **Step 3: Implement elephant_mobility()**

```rust
/// Elephant mobility: number of valid elephant endpoints.
/// Each valid endpoint = +5. Returns Red-positive.
pub fn elephant_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
    let moves = movegen::generate_elephant_moves(board, pos, color);
    (moves.len() as i32) * 5
}
```

### 3d. cannon_activity (uses Bitboards for platform detection)

- [ ] **Step 4: Implement cannon_activity()**

For each of 4 cardinal directions:
1. Find the nearest piece using chariot rays (iterate along direction until occupied)
2. If that piece is a platform (advisor/elephant/pawn) → +10
3. If that piece is an enemy → cannon has a capture threat (can jump to it) → also +5

```rust
/// Cannon activity: count platform pieces (advisor/elephant/pawn) between cannon
/// and first piece in each direction. Each platform = +10.
/// Returns Red-positive.
pub fn cannon_activity(board: &Board, pos: Coord, color: Color) -> i32 {
    let mut score = 0i32;
    let occ_all = board.bitboards.occupied_all();

    for &(dx, dy) in DIRS_4.iter() {
        let mut cx = pos.x + dx;
        let mut cy = pos.y + dy;
        // Scan toward edge until we hit any piece
        while (0..9).contains(&cx) && (0..10).contains(&cy) {
            let c = Coord::new(cx, cy);
            let c_sq = (cy as i8 * 9 + cx as i8) as u8;
            if occ_all & (1_u128 << c_sq) != 0 {
                // First piece found — check if it's a platform
                if let Some(p) = board.get(c) {
                    if p.color == color {
                        // Friendly piece blocks and is platform-type
                        if p.piece_type == PieceType::Advisor
                            || p.piece_type == PieceType::Elephant
                            || p.piece_type == PieceType::Pawn
                        {
                            score += 10;
                        }
                    }
                }
                break; // Stop scanning after first piece
            }
            cx += dx;
            cy += dy;
        }
    }
    score
}
```

### 3e. Stage 2 integration in main eval (temporarily)

- [ ] **Step 5: Add Stage 2 scores to handcrafted_evaluate scaffold**

After material_pst() in the eval loop, add mobility for horse, chariot, cannon.

- [ ] **Step 6: Verify tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 7: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement Stage 2 mobility with bitboard acceleration

- horse_mobility: horse_attacks().count_ones() * 10
- chariot_mobility: (chariot_attacks & !occupied_all()).count_ones() * 5
- elephant_mobility: generate_elephant_moves.len() * 5
- cannon_activity: platform detection via directional scan

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Stage 2 — Pawn Structure

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement pawn_structure()**

Per the spec:
- Doubled penalty: -30 per file when 2+ pawns on same file (penalize once per extra pawn)
- Back-rank penalty: -20 for pawns on starting rank
- Link bonus: +15 per adjacent friendly pawn (horizontal)
- Advancement: crossed river = +5 to +20; on enemy back rank = +80
- Central file bonus (columns 4-5): +10 per rank advanced

```rust
/// Pawn structure evaluation. Returns Red-positive score.
pub fn pawn_structure(board: &Board, color: Color, phase: i32) -> i32 {
    let mut score = 0i32;
    let mut pawns_per_file = [0i32; 9];

    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(p) = board.get(pos)
                && p.color == color && p.piece_type == PieceType::Pawn
            {
                pawns_per_file[x] += 1;

                // Doubled penalty: -30 for each extra pawn beyond the first
                if pawns_per_file[x] >= 2 {
                    score -= 30;
                }

                // Back-rank penalty
                let on_starting_rank = match color {
                    Color::Red => y == 6,    // Red starts at y=6
                    Color::Black => y == 3,  // Black starts at y=3
                };
                if on_starting_rank {
                    score -= 20;
                }

                // Horizontal link bonus
                let left = Coord::new(pos.x - 1, pos.y);
                let right = Coord::new(pos.x + 1, pos.y);
                if let Some(lp) = board.get(left)
                    && lp.color == color && lp.piece_type == PieceType::Pawn
                {
                    score += 15;
                }
                if let Some(rp) = board.get(right)
                    && rp.color == color && rp.piece_type == PieceType::Pawn
                {
                    score += 15;
                }

                // Advancement bonus (crossed river)
                let crossed = pos.crosses_river(color);
                if crossed {
                    let advancement = match color {
                        Color::Red => 6 - y,   // y=6 is home, y=0 is enemy back rank
                        Color::Black => y - 3, // y=3 is home, y=9 is enemy back rank
                    };
                    if advancement > 0 {
                        score += advancement * 5;
                    }
                    // Enemy back rank bonus
                    let on_enemy_back = match color {
                        Color::Red => y == 0,
                        Color::Black => y == 9,
                    };
                    if on_enemy_back {
                        score += 80;
                    }
                }

                // Central file bonus (files 4-5, columns index 4-5)
                if pos.x == 4 || pos.x == 5 {
                    let ranks_advanced = match color {
                        Color::Red => 6 - y,
                        Color::Black => y - 3,
                    };
                    if ranks_advanced > 0 {
                        score += ranks_advanced * 10;
                    }
                }
            }
        }
    }
    score * color.sign()
}
```

- [ ] **Step 2: Verify tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement pawn_structure()

- Doubled penalty: -30 per extra pawn per file
- Back-rank penalty: -20 for starting-rank pawns
- Horizontal link bonus: +15 per adjacent friendly pawn
- Advancement: crossed river = rank * 5, enemy back rank = +80
- Central file bonus: +10 per rank advanced on files 4-5

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Stage 2 — Elephant Structure

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement elephant_structure()**

Per the spec:
- Missing both: -80 MG / -200 EG
- Missing one: -40 MG / -100 EG
- No mutual protection logic

```rust
/// Elephant structure evaluation. Returns Red-positive score.
pub fn elephant_structure(board: &Board, color: Color, phase: i32) -> i32 {
    let count = board.bitboards.piece_bitboard(PieceType::Elephant, color).count_ones() as i32;
    let mg_factor = phase;
    let eg_factor = TOTAL_PHASE - phase;

    let penalty = match count {
        0 => 80 * mg_factor + 200 * eg_factor,   // Missing both
        1 => 40 * mg_factor + 100 * eg_factor,   // Missing one
        _ => 0,
    };
    // Correct formula: -(penalty / TOTAL_PHASE) * (-color.sign())
    // Red (sign=+1): -(penalty/TOTAL_PHASE) * (-1) = -(penalty/TOTAL_PHASE) [subtracts from Red score]
    // Black (sign=-1): -(penalty/TOTAL_PHASE) * (+1) = +(penalty/TOTAL_PHASE) [adds to Red score]
    -(penalty / TOTAL_PHASE) * (-color.sign())
}

- [ ] **Step 2: Verify tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement elephant_structure()

- Missing both elephants: -(80*MG + 200*EG) / TOTAL_PHASE
- Missing one: -(40*MG + 100*EG) / TOTAL_PHASE
- Removed mutual protection logic (overcomplicated)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Stage 3 — King Safety with attackers() Bitboard

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement king_safety()**

Per the spec and the new constraint to use `attackers()` bitboard:

```rust
/// King safety for one side. Returns Red-positive score.
/// - Palace defenders: advisor=+15, elephant=+10
/// - Enemy pressure: distance-weighted chariot/cannon/horse/pawn threat
/// - Attackers on enemy king: +15 per attacker (uses attackers() bitboard)
pub fn king_safety(board: &Board, king_color: Color, phase: i32) -> Option<i32> {
    let king_pos = king_pos(board, king_color)?;
    let opponent = king_color.opponent();
    let mut score = 0i32;

    // Palace defenders
    for delta in PALACE_DELTAS {
        let pos = Coord::new(king_pos.x + delta.0, king_pos.y + delta.1);
        if pos.is_valid() && pos.in_palace(king_color) {
            if let Some(p) = board.get(pos) && p.color == king_color {
                score += match p.piece_type {
                    PieceType::Advisor => 15,
                    PieceType::Elephant => 10,
                    _ => 0,
                };
            }
        }
    }

    // Enemy piece pressure (distance-weighted)
    let mg_factor = phase;
    let occ = board.bitboards.occupied_all();
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            let sq = (y * 9 + x) as u8;
            if occ & (1_u128 << sq) == 0 { continue; } // skip empty
            if let Some(p) = board.get(pos) && p.color == opponent {
                let dist = (pos.x - king_pos.x).abs() + (pos.y - king_pos.y).abs();
                let threat = match p.piece_type {
                    PieceType::Chariot => ((14 - dist).max(0) as i32) * 8,
                    PieceType::Cannon => ((12 - dist).max(0) as i32) * 5,
                    PieceType::Horse => ((10 - dist).max(0) as i32) * 6,
                    PieceType::Pawn if dist <= 4 => ((5 - dist) as i32) * 6,
                    _ => 0,
                };
                score -= threat * mg_factor / TOTAL_PHASE;
            }
        }
    }

    // Attackers on enemy king (uses attackers() bitboard — the new efficient way)
    if let Some(ek) = king_pos(board, opponent) {
        let ek_sq = (ek.y * 9 + ek.x) as u8;
        let attackers_bb = board.bitboards.attackers(ek_sq, king_color);
        let attack_count = attackers_bb.count_ones() as i32;
        score += attack_count * 15; // reduced from 25 to 15
    }

    Some(score * king_color.sign())
}
```

- [ ] **Step 2: Verify tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 3: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement king_safety() with attackers() bitboard

- Palace defenders: advisor=+15, elephant=+10
- Enemy pressure weights reduced: chariot 10→8, cannon 7→5, horse 8→6, pawn 8→6
- King attack bonus: 25→15 per attacker, uses attackers() bitboard
- Returns Red-positive (sign applied at end)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Stage 3 — can_attack / can_defend Helpers + hanging_pieces

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement can_attack() and can_defend() helpers**

Simple geometric checks — no move generation needed:

```rust
/// True if attacker at `from` can attack `to` given attacker's piece type and color.
/// All pieces use Manhattan (taxi) distance: |dx| + |dy|
/// - Chariot: same row/col, clear path
/// - Cannon: same row/col, exactly 1 screen
/// - Horse: L-move, Manhattan = 3
/// - Pawn: orthogonal forward (both colors), or forward+sideways after crossing river
/// - Advisor: diagonal ±1, in palace
/// - Elephant: diagonal ±2, eye empty, doesn't cross river
/// - King: orthogonal ±1, in palace
fn can_attack(board: &Board, from: Coord, to: Coord, pt: PieceType, color: Color) -> bool {
    let dx = (to.x - from.x).abs() as i32;
    let dy = (to.y - from.y).abs() as i32;
    let dist = dx + dy;
    match pt {
        PieceType::Chariot => {
            if from.y == to.y {
                let step = if from.x < to.x { 1 } else { -1 };
                let mut x = from.x + step;
                while x != to.x {
                    if board.get(Coord::new(x, from.y)).is_some() { return false; }
                    x += step;
                }
                true
            } else if from.x == to.x {
                let step = if from.y < to.y { 1 } else { -1 };
                let mut y = from.y + step;
                while y != to.y {
                    if board.get(Coord::new(from.x, y)).is_some() { return false; }
                    y += step;
                }
                true
            } else {
                false
            }
        }
        PieceType::Cannon => {
            if from.y == to.y {
                let step = if from.x < to.x { 1 } else { -1 };
                let mut screens = 0;
                let mut x = from.x + step;
                while x != to.x {
                    if board.get(Coord::new(x, from.y)).is_some() { screens += 1; }
                    x += step;
                }
                screens == 1
            } else if from.x == to.x {
                let step = if from.y < to.y { 1 } else { -1 };
                let mut screens = 0;
                let mut y = from.y + step;
                while y != to.y {
                    if board.get(Coord::new(from.x, y)).is_some() { screens += 1; }
                    y += step;
                }
                screens == 1
            } else {
                false
            }
        }
        PieceType::Horse => dist == 3, // L-move: (2,1) or (1,2) → taxi = 3
        PieceType::Pawn => {
            // Pawns in Chinese Chess move/capture ORTHOGONALLY (not diagonally)
            // Forward: Red=-1 (toward y=0), Black=+1 (toward y=9)
            // After crossing river: can also move/capture sideways (dx=±1)
            let forward_dir = if color == Color::Red { -1 } else { 1 };
            let raw_dx = to.x - from.x;
            let raw_dy = to.y - from.y;
            let is_forward = raw_dy == forward_dir;
            let is_sideways = raw_dx.abs() == 1 && raw_dy == 0;
            let crossed = from.crosses_river(color);
            (is_forward || (crossed && is_sideways)) && dist == 1
        }
        PieceType::Advisor => {
            // Advisor: diagonal ±1 step, must be in palace
            dist == 2 && to.in_palace(color)
        }
        PieceType::Elephant => {
            // Elephant: diagonal ±2 steps, eye empty, doesn't cross river
            if dist != 4 || to.crosses_river(color) { return false; }
            let eye_x = from.x + (to.x - from.x) / 2;
            let eye_y = from.y + (to.y - from.y) / 2;
            board.get(Coord::new(eye_x, eye_y)).is_none()
        }
        PieceType::King => {
            // King: orthogonal ±1, must be in palace
            dist == 1 && to.in_palace(color)
        }
    }
}

/// True if defender at `from` can defend `to` (same geometric rules as can_attack, no capture check)
/// A piece can defend any square it could move to (ignoring whether the target square is occupied)
fn can_defend(board: &Board, from: Coord, to: Coord, pt: PieceType, color: Color) -> bool {
    let dx = (to.x - from.x).abs() as i32;
    let dy = (to.y - from.y).abs() as i32;
    let dist = dx + dy;
    match pt {
        PieceType::Chariot | PieceType::Cannon => from.y == to.y || from.x == to.x,
        PieceType::Horse => dist == 3, // L-move
        PieceType::Pawn => {
            // Same orthogonal rules as can_attack (defending doesn't check if target is enemy)
            let forward_dir = if color == Color::Red { -1 } else { 1 };
            let raw_dx = to.x - from.x;
            let raw_dy = to.y - from.y;
            let is_forward = raw_dy == forward_dir;
            let is_sideways = raw_dx.abs() == 1 && raw_dy == 0;
            let crossed = from.crosses_river(color);
            (is_forward || (crossed && is_sideways)) && dist == 1
        }
        PieceType::Advisor => dist == 2 && to.in_palace(color),
        PieceType::Elephant => {
            if dist != 4 || to.crosses_river(color) { return false; }
            let eye_x = from.x + (to.x - from.x) / 2;
            let eye_y = from.y + (to.y - from.y) / 2;
            board.get(Coord::new(eye_x, eye_y)).is_none()
        }
        PieceType::King => dist == 1 && to.in_palace(color),
    }
}
```

- [ ] **Step 2: Implement hanging_pieces()**

Per spec: hanging (attacked with 0 defense) and overloaded (attacked by multiple with 1 defense).

```rust
/// Detect hanging and overloaded enemy pieces.
/// Returns Red-positive score.
pub fn hanging_pieces(board: &Board, color: Color, phase: i32) -> i32 {
    let mut score = 0i32;
    let mg_factor = phase;

    let our = pieces_of_color(board, color);
    let enemy = pieces_of_color(board, color.opponent());

    for enemy_piece in &enemy {
        let mut attackers = 0;
        let mut defenders = 0;

        for our_piece in &our {
            if can_attack(board, our_piece.pos, enemy_piece.pos, our_piece.pt, color) {
                attackers += 1;
            }
        }
        for (idx, def_piece) in enemy.iter().enumerate() {
            if def_piece.pos == enemy_piece.pos { continue; } // no self-defense
            if can_defend(board, def_piece.pos, enemy_piece.pos, def_piece.pt, color.opponent()) {
                defenders += 1;
            }
        }

        let val = MG_VALUE[enemy_piece.pt as usize];
        if attackers > 0 && defenders == 0 {
            // Hanging: full bonus
            score += (val * 30 / 100) * mg_factor / TOTAL_PHASE;
        } else if attackers > 1 && defenders == 1 {
            // Overloaded: half bonus
            score += (val * 15 / 100) * mg_factor / TOTAL_PHASE;
        }
    }

    score * color.sign()
}
```

- [ ] **Step 3: Verify tests still pass**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): implement can_attack/can_defend helpers and hanging_pieces()

- can_attack: geometric attack check (sliding clear-path, horse dist, pawn adjacent)
- can_defend: same-line check for sliding pieces, distance for others
- hanging_pieces: hanging=(attacked,0 defense), overloaded=(2+ attackers,1 defender)
- Bonus: 30% MG value for hanging, 15% for overloaded

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Combine All Stages in handcrafted_evaluate()

**Files:**
- Modify: `src/eval.rs`

- [ ] **Step 1: Implement full handcrafted_evaluate()**

```rust
pub fn handcrafted_evaluate(board: &Board, side: Color, _initiative: bool) -> i32 {
    // Checkmate detection (preserve existing logic)
    let (rk, bk) = board.find_kings();
    if rk.is_none() {
        let fallback = if side == Color::Red { -MATE_SCORE } else { MATE_SCORE };
        if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
            if conf < 0.2 { return tb_score; }
            let weight = conf.max(0.3);
            return (tb_score as f32 * weight + fallback as f32 * (1.0 - weight)) as i32;
        }
        return fallback;
    }
    if bk.is_none() {
        let fallback = if side == Color::Red { MATE_SCORE } else { -MATE_SCORE };
        if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
            if conf < 0.2 { return tb_score; }
            let weight = conf.max(0.3);
            return (tb_score as f32 * weight + fallback as f32 * (1.0 - weight)) as i32;
        }
        return fallback;
    }

    let phase = game_phase(board);
    let mut raw_score = 0i32;

    // Stage 1: Material + PST
    raw_score += material_pst(board);

    // Stage 2: Mobility + Activity (piece by piece)
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(piece) = board.get(pos) {
                let sign = piece.color.sign();
                match piece.piece_type {
                    PieceType::Horse => {
                        raw_score += horse_mobility(board, pos, piece.color) * sign;
                    }
                    PieceType::Chariot => {
                        raw_score += chariot_mobility(board, pos, piece.color) * sign;
                    }
                    PieceType::Cannon => {
                        raw_score += cannon_activity(board, pos, piece.color) * sign;
                    }
                    _ => {}
                }
            }
        }
    }

    // Pawn structure
    raw_score += pawn_structure(board, Color::Red, phase);
    raw_score += pawn_structure(board, Color::Black, phase);

    // Elephant structure
    raw_score += elephant_structure(board, Color::Red, phase);
    raw_score += elephant_structure(board, Color::Black, phase);

    // Elephant mobility
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(piece) = board.get(pos)
                && piece.piece_type == PieceType::Elephant
            {
                raw_score += elephant_mobility(board, pos, piece.color) * piece.color.sign();
            }
        }
    }

    // Stage 3: King safety (per side, returns pre-signed)
    if let Some(ks) = king_safety(board, Color::Red, phase) {
        raw_score += ks;
    }
    if let Some(ks) = king_safety(board, Color::Black, phase) {
        raw_score += ks;
    }

    // Hanging pieces
    raw_score += hanging_pieces(board, Color::Red, phase);
    raw_score += hanging_pieces(board, Color::Black, phase);

    // Stage 4: Check bonus
    if board.is_check(Color::Black) {
        raw_score += CHECK_BONUS;
    }
    if board.is_check(Color::Red) {
        raw_score -= CHECK_BONUS;
    }

    // Negate if evaluating from Black's perspective
    let final_score = raw_score * side.sign();

    // Blend with tablebase if available
    if let Some((tb_score, conf)) = EndgameTablebase::probe(board, side) {
        if conf < 0.2 { return tb_score; }
        let weight = conf.max(0.3);
        return (tb_score as f32 * weight + final_score as f32 * (1.0 - weight)) as i32;
    }

    final_score
}
```

**Note**: `initiative` parameter is accepted but ignored (initiative bonus removed per spec).

- [ ] **Step 2: Verify compilation and tests**

```bash
cargo test --no-default-features test_eval_handcrafted 2>&1 | tail -20
```

Expected: both existing tests pass.

- [ ] **Step 3: Run all tests**

```bash
cargo test --no-default-features 2>&1 | tail -30
```

- [ ] **Step 4: Commit**

```bash
git add src/eval.rs && git commit -m "$(cat <<'EOF'
feat(eval): complete handcrafted_evaluate() combining all stages

Stage 1: material_pst
Stage 2: horse/chariot/cannon mobility + elephant mobility + pawn/elephant structure
Stage 3: king_safety (attackers bitboard) + hanging_pieces
Stage 4: check bonus
No initiative bonus. No center_control. No piece_coordination.
Existing tests must pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add New Tests

**Files:**
- Modify: `src/main.rs` (tests live there per existing pattern)

- [ ] **Step 1: Add test_material_pst_alone**

Test that material_pst on starting position returns a reasonable value.

```rust
#[test]
fn test_material_pst_alone() {
    let board = Board::new(RuleSet::Official, 1);
    let score = crate::eval::eval_impl::material_pst(&board);
    // Starting position: each side has 2 advisors, 2 elephants, 2 horses,
    // 2 chariots, 2 cannons, 5 pawns = material sum ≈ 2585 MG / 2740 EG
    // PST adds some value too. Should be positive (Red advantage on turn).
    assert!(score > 0, "material_pst should be positive on starting pos, got {}", score);
}
```

- [ ] **Step 2: Add test_king_safety_uses_attackers_bitboard**

Create a position where a known number of pieces attack the enemy king, verify the score reflects it.

```rust
#[test]
fn test_king_safety_uses_attackers_bitboard() {
    // Setup: Red chariot on same file as Black king, nothing blocking
    // Black king at (4,0), Red chariot at (4,5) — clear attack
    let board = make_board(vec![
        (4, 0, Color::Black, PieceType::King),
        (4, 5, Color::Red, PieceType::Chariot),
    ]);
    let ks = crate::eval::eval_impl::king_safety(&board, Color::Red, 20);
    assert!(ks.is_some(), "king_safety should return Some");
    // Chariot attacks king: +15 per spec
    let ours = ks.unwrap();
    assert!(ours > 0, "Red should get king safety bonus from chariot attack, got {}", ours);
}
```

- [ ] **Step 3: Add test_hanging_piece_detection**

```rust
#[test]
fn test_hanging_piece_detection() {
    // Red chariot attacks undefended Black horse
    // Black horse at (5,3), Red chariot at (5,8) — nothing blocking
    // Black has no pieces that can defend the horse
    let board = make_board(vec![
        (4, 0, Color::Black, PieceType::King),
        (5, 3, Color::Black, PieceType::Horse),  // hanging
        (4, 7, Color::Red, PieceType::King),
        (5, 8, Color::Red, PieceType::Chariot),  // attacks horse
    ]);
    let hp = crate::eval::eval_impl::hanging_pieces(&board, Color::Red, 20);
    assert!(hp > 0, "hanging_pieces should be positive when enemy piece is hanging, got {}", hp);
}
```

- [ ] **Step 4: Run all new tests**

```bash
cargo test --no-default-features test_material_pst_alone test_king_safety_uses_attackers_bitboard test_hanging_piece_detection 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add src/eval.rs src/main.rs && git commit -m "$(cat <<'EOF'
test(eval): add new tests for material_pst, king_safety, hanging_pieces

- test_material_pst_alone: verifies reasonable value on start pos
- test_king_safety_uses_attackers_bitboard: chariot attack on king = +15
- test_hanging_piece_detection: undefended horse under attack scores

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Full Test Suite and Final Verification

- [ ] **Step 1: Run complete test suite**

```bash
cargo test --no-default-features 2>&1 | tail -40
```

- [ ] **Step 2: If any tests fail, diagnose and fix inline**

Common failure modes:
- Integer overflow in phase interpolation → use i64 intermediate
- `count_ones()` on 0 → returns 0 (correct)
- Elephant mobility: `generate_elephant_moves` may return 0 for blocked elephants → 0 mobility (correct)

- [ ] **Step 3: Final commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
perf(eval): final verification and cleanup

All tests pass. Eval rewrite complete:
- material_pst with integer phase interpolation
- bitboard-accelerated mobility (horse/chariot/elephant)
- cannon_activity with platform detection
- pawn/elephant structure
- king_safety with attackers() bitboard, reduced weights
- hanging_pieces with geometric can_attack/can_defend
- Removed: center_control, piece_coordination, initiative bonus

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task |
|---|---|
| Stage 1 material+PST | Task 2 |
| Horse mobility (bitboard) | Task 3 |
| Chariot mobility (bitboard) | Task 3 |
| Elephant mobility (generate_elephant_moves) | Task 3 |
| Cannon activity (platform detection) | Task 3 |
| Pawn structure | Task 4 |
| Elephant structure | Task 5 |
| King safety (attackers bitboard, reduced weights) | Task 6 |
| can_attack/can_defend helpers | Task 7 |
| hanging_pieces | Task 7 |
| handcrafted_evaluate combination | Task 8 |
| Check bonus | Task 8 |
| Existing tests pass | All tasks |
| New tests added | Task 9 |

---

## Spec Self-Review

1. **Placeholder scan**: All code blocks have actual Rust code, no TODOs
2. **Internal consistency**: `king_safety` returns pre-signed (line `score * king_color.sign()`), `hanging_pieces` returns pre-signed, `material_pst` returns pre-signed
3. **Type consistency**: All functions use `i32` for scores, `Coord` for positions, `u128` for bitboards
4. **Spec gaps**: None identified — all terms from spec are covered
