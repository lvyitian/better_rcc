# Handcrafted Evaluation Rewrite Design

## Overview

A complete rewrite of `handcrafted_evaluate` in `src/eval.rs` with three goals:
1. **Maintainability** — single-pass architecture, one sign convention, no overlapping terms
2. **Evaluation quality** — clean, tunable terms without noise from overlapping bonuses
3. **Simplicity** — remove initiative bonus; reduce king attack weight; drop `center_control`, `piece_coordination`, `elephant_structure` mutual protection

**Compatibility contract:** existing tests (`test_eval_handcrafted_returns_reasonable_values`, `test_eval_handcrafted_symmetry_red_black`) must pass.

---

## Sign Convention

All intermediate scoring functions return **positive = advantage for Red**.

The final `handcrafted_evaluate(board, side, initiative)`:
1. Computes `raw_score` with Red = positive
2. Negates if `side == Black`
3. Does NOT add initiative bonus (removed)

---

## Stage 1: Material + PST

### 1a. Material Value

Piece values (MG / EG), interpolated by game phase:

| Piece | MG | EG |
|---|---|---|
| King | 10000 | 10000 |
| Advisor | 135 | 140 |
| Elephant | 105 | 100 |
| Pawn | 80 | 200 |
| Horse | 350 | 450 |
| Cannon | 500 | 380 |
| Chariot | 650 | 700 |

### 1b. Piece-Square Tables (PST)

Keep existing tables unchanged:
- `MG_PST_KING`, `EG_PST_KING` — palace position bonuses
- `MG_PST_ADVISOR` — palace corner/diagonal preferences
- `MG_PST_ELEPHANT` — back-rank preference, river penalty
- `MG_PST_HORSE` — center advancement
- `MG_PST_CANNON` — center + palace entrance elevation
- `MG_PST_CHARIOT` — open line centrality
- `MG_PST_PAWN`, `EG_PST_PAWN` — forward advancement, river crossing

### 1c. Phase Interpolation

```
t = phase / TOTAL_PHASE   // 0 = endgame, 1 = midgame
value = mg * t + eg * (1 - t)
```

### 1d. Implementation

Single function `material_pst(board) -> i32` that iterates all pieces once:
```rust
pub fn material_pst(board: &Board) -> i32 {
    let phase = game_phase(board);
    let t = phase as f32 / TOTAL_PHASE as f32;
    let mut score = 0i32;

    for y in 0..10 {
        for x in 0..9 {
            if let Some(piece) = board.get(Coord::new(x as i8, y as i8)) {
                let mg = MG_VALUE[piece.piece_type as usize];
                let eg = EG_VALUE[piece.piece_type as usize];
                let val = (mg as f32 * t + eg as f32 * (1.0 - t)) as i32;
                score += val * piece.color.sign();

                let pst = pst_val(piece.piece_type, piece.color, x, y);
                score += pst;
            }
        }
    }
    score
}
```

---

## Stage 2: Mobility + Activity

Each piece type gets a single mobility/activity score. No double-counting.

### Horse Mobility
- Count valid knight jumps (empty horse-head position)
- Each valid jump = +10
- Single function `horse_mobility(board, pos, color) -> i32`

### Chariot Mobility
- Empty squares reachable in 4 cardinal directions (no capture check)
- Each empty square = +5
- Single function `chariot_mobility(board, pos, color) -> i32`

### Cannon Activity
- For each of 4 cardinal directions: count platform pieces (advisor, elephant, pawn) between cannon and first enemy piece
- Each platform = +10 (cannon with screen is dangerous)
- Single function `cannon_activity(board, pos, color) -> i32`

### Pawn Structure
- **Doubled penalty**: -30 per file with 2+ pawns (penalize once per extra pawn)
- **Back-rank penalty**: -20 for pawns on starting rank (not crossed river)
- **Link bonus**: +15 per adjacent friendly pawn (horizontal linking)
- **Advancement bonus**:
  - Crossed river: +5 to +20 based on how far advanced
  - On enemy back rank: +80 (threat to promote)
- **Central file bonus**: +10 per rank advanced on files 4-5 (columns 4-5)
- Single function `pawn_structure(board, color, phase) -> i32`

### Elephant Structure
- **Missing both elephants**: -80 MG / -200 EG
- **Missing one elephant**: -40 MG / -100 EG
- Single function `elephant_structure(board, color, phase) -> i32`
- **Note**: mutual protection logic removed (overcomplicated, noisy)

### Elephant Mobility (NEW — replaces confusion in old code)
- Count valid elephant jumps (blocked by river, blocked by friendly pieces)
- Use `movegen::generate_elephant_moves` — already exists
- Each valid endpoint = +5

### Removed from Stage 2
- `center_control` — overlaps with chariot/cannon mobility; chariot already scores for open lines near center
- `piece_coordination` — horse+chariot combo was double-counted by mobility terms

---

## Stage 3: King Safety + Threats

### King Safety (per side)
```rust
fn king_safety(board: &Board, king_color: Color, phase: i32) -> i32 {
    let king_pos = board.find_king(king_color)?;
    let opponent = king_color.opponent();
    let mut score = 0i32;

    // Palace defenders: advisors +15 each, elephants +10 each
    for delta in PALACE_DELTAS {
        let pos = king_pos + delta;
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

    // Enemy piece pressure (distance-weighted threat)
    let mg_factor = phase;
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(p) = board.get(pos) && p.color == opponent {
                let dist = (pos.x - king_pos.x).abs() + (pos.y - king_pos.y).abs();
                let threat = match p.piece_type {
                    PieceType::Chariot => (14 - dist).max(0) as i32 * 8,
                    PieceType::Cannon => (12 - dist).max(0) as i32 * 5,
                    PieceType::Horse => (10 - dist).max(0) as i32 * 6,
                    PieceType::Pawn if dist <= 4 => (5 - dist) as i32 * 6,
                    _ => 0,
                };
                score -= threat * mg_factor / TOTAL_PHASE;
            }
        }
    }

    // Attackers on enemy king (use attackers() bitboard)
    let enemy_king_pos = board.find_king(opponent);
    if let Some(ek) = enemy_king_pos {
        let ek_sq = (ek.y * 9 + ek.x) as u8;
        let attackers = board.bitboards.attackers(ek_sq, king_color);
        let attack_count = attackers.count_ones() as i32;
        // Reduced from +25 to +15 per attacker
        score += attack_count * 15;
    }

    score * king_color.sign()
}
```

**Changes from old `king_safety`:**
- Reduced per-threat weights (chariot: 10→8, cannon: 7→5, horse: 8→6, pawn: 8→6)
- Reduced king attack bonus: 25→15 per attacker
- Returns Red-positive (removed internal sign flip)

### Hanging / Overloaded Piece Detection
```rust
fn hanging_pieces(board: &Board, color: Color, phase: i32) -> i32 {
    let mut score = 0i32;
    let mg_factor = phase;
    let eg_factor = TOTAL_PHASE - phase;

    let our_pieces: Vec<_> = board.pieces_of_color(color).collect();
    let enemy_pieces: Vec<_> = board.pieces_of_color(color.opponent()).collect();

    for enemy in &enemy_pieces {
        let mut attackers = 0;
        let mut defenders = 0;

        for our in &our_pieces {
            if can_attack(board, our.pos, enemy.pos, our.pt) {
                attackers += 1;
            }
        }
        for (idx, def) in enemy_pieces.iter().enumerate() {
            if def.pos == enemy.pos { continue; } // exclude self
            if can_defend(board, def.pos, enemy.pos, def.pt) {
                defenders += 1;
            }
        }

        if attackers > 0 && defenders == 0 {
            // Hanging: full penalty
            let val = MG_VALUE[enemy.pt as usize];
            score += (val * 30 / 100) * mg_factor / TOTAL_PHASE;
        } else if attackers > 1 && defenders == 1 {
            // Overloaded: partial bonus
            let val = MG_VALUE[enemy.pt as usize];
            score += (val * 15 / 100) * mg_factor / TOTAL_PHASE;
        }
    }

    score * color.sign()
}
```

**Note**: `can_attack` and `can_defend` are simple geometric checks (same row/col for sliding, distance for non-sliding). No move generation needed.

---

## Stage 4: Check Bonus

```rust
if board.is_check(Color::Black) { raw_score += CHECK_BONUS; }
if board.is_check(Color::Red) { raw_score -= CHECK_BONUS; }
```

CHECK_BONUS stays at existing value (keep it tunable via const).

---

## What Gets Removed (Complete List)

| Function / Feature | Reason |
|---|---|
| `center_control` | Overlaps with chariot/cannon mobility; not independently tunable |
| `piece_coordination` | Horse+chariot combo was counted twice (mobility + coordination) |
| `attack_rewards` (old) | Mixed hanging pieces, central attacks, king threats — unmaintainable |
| `elephant_structure` mutual protection | Overcomplicated; added noise |
| `cannon_support` | Replaced by `cannon_activity` |
| Initiative bonus (`+20 * side.sign()`) | Removed per decision |
| `king_safety` per-attacker bonus of 25 | Reduced to 15 |
| Old threat weights (chariot 10, cannon 7, horse 8, pawn 8) | Reduced proportionally |

---

## Removed Constants

- `CHECK_BONUS` — keep (used in Stage 4)
- `MATE_SCORE` — keep (used for checkmate detection)

---

## File Structure

```rust
// src/eval.rs
pub mod eval_impl {
    // Constants (MG/EG values, PST tables, phase weights)
    // game_phase()
    // pst_val()

    // Stage 1
    // material_pst()

    // Stage 2
    // horse_mobility()
    // chariot_mobility()
    // cannon_activity()
    // pawn_structure()
    // elephant_structure()

    // Stage 3
    // king_safety()
    // can_attack()      // helper: geometric check for attack
    // can_defend()      // helper: geometric check for defense
    // hanging_pieces()

    // Main entry point
    // handcrafted_evaluate(board, side, initiative)
}
```

**Note**: The `initiative` parameter is kept in the function signature for backward compatibility, but the body ignores it (no initiative bonus).

---

## NNUE Alignment (Read-Only)

This design does NOT change NNUE input features. The handcrafted eval terms are designed to be **comparable to** NNUE features for quality testing (e.g., "NNUE eval vs. handcrafted eval on the same positions") but they need not map 1:1.

---

## Testing

### Existing Tests (Must Pass)
- `test_eval_handcrafted_returns_reasonable_values` — Red eval > 0, Black eval < 0 on starting position
- `test_eval_handcrafted_symmetry_red_black` — evaluations are negated correctly

### New Tests to Add
1. `test_material_pst_alone` — verify material_pst returns expected range on starting position
2. `test_king_safety_uses_attackers_bitboard` — verify king attack bonus uses `attackers()` not adjacency
3. `test_no_negative_material` — total material (ignoring PST) should never go negative midgame
4. `test_hanging_piece_detection` — simple position with one hanging piece scores correctly
5. `test_phase_interpolation` — verify EG pawn value surge (80→200) affects eval

---

## Implementation Order

1. Implement Stage 1 (`material_pst`) — verify existing tests pass
2. Implement Stage 2 mobility functions — verify no regressions
3. Implement Stage 3 (`king_safety`, `hanging_pieces`) — verify king attack bonus works
4. Implement `handcrafted_evaluate` combining all stages
5. Add new tests
6. Run full test suite
7. Optional: tuning pass (adjust weights if eval quality is off)
