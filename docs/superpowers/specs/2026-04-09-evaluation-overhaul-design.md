# Evaluation Overhaul Design

> **Goal:** Make the engine play aggressively by significantly increasing offensive signals — attack rewards, check pressure, mobility, pawn advancement, and initiative.

**Date:** 2026-04-09

## 1. Architecture

### Evaluation Flow
```
evaluate(board, side)
├── material_pst      // base value: piece values + PST
├── mobility         // all pieces scored equivalently
├── king_safety      // defense + own attack pressure
├── attack_rewards  // NEW: hanging pieces, overload, central/king attacks
├── check_bonus      // 250 (was 50)
├── initiative_bonus // NEW: +20 if last move was aggressive
├── pawn_structure  // added: advancement rewards
├── center_control  // increased weight
├── piece_coordination // enhanced offensive bonus
└── elephant_structure // added: mutual pair protection
```

### File: `src/main.rs`
- Modify the `eval` module (lines ~2369-2986)
- Add `attack_rewards()` function after `center_control()`
- Add `initiative_bonus()` tracking in search context

## 2. Mobility (Normalized)

**Current (inconsistent):**
- Horse: ×5
- Chariot: ×1
- Cannon support: ×1

**Proposed (normalized):**
- All pieces: ×5 score per move
- Horse mobility: 0-8 jumps → 0-40 score
- Chariot mobility: open line squares ×5
- Cannon support: screen pieces ×10

## 3. King Safety (Expanded)

### Own Attack Pressure (near enemy king)
| Piece Type | Bonus per piece attacking king's adjacent squares |
|------------|-------------------------------------------------|
| Chariot/Cannon | +15 |
| Horse | +10 |
| Pawn (after crossing river) | +5 |

### King Attack Bonus
- Any piece directly threatening enemy king square: **+25 per piece**

### Defense Bonus (existing, unchanged)
- Advisor protecting king: +25
- Elephant protecting king: +15
- Other piece: +5

### Enemy Threat Distance (existing, unchanged)
- Chariot: (14 - dist) × 10
- Cannon: (12 - dist) × 7
- Horse: (10 - dist) × 8
- Pawn (dist ≤ 4): (5 - dist) × 8

## 4. Attack Rewards (New Function)

New function: `fn attack_rewards(board: &Board, color: Color, phase: i32) -> i32`

### 4.1 Hanging Piece Bonus: +30 per piece
An enemy piece is "hanging" when:
- It has 0 defending pieces, OR
- It has 1 defender but the defender is also defending something higher-value

Detection: For each enemy piece, count attack threats vs defense count. If attack > defense and no higher-value protector exists, it's hanging.

### 4.2 Overload Penalty: -15 per piece
A friendly piece is overloaded when it is the sole defender of a piece that is also attacked. This creates tactical vulnerability.

### 4.3 Central Attack Bonus: +10 per piece
Each of your pieces attacking a core area square (x ∈ [3,5], y ∈ [4,5]) gets +10.

### 4.4 King Attack Bonus: +25 per attacking piece
Any piece whose attack target list includes the enemy king square gets +25.

### Phase Scaling
- `mg_factor = phase`, `eg_factor = TOTAL_PHASE - phase`
- Hanging piece bonus: (30 × mg + 20 × eg) / TOTAL_PHASE
- Central attack: (10 × mg + 5 × eg) / TOTAL_PHASE
- King attack: (25 × mg + 15 × eg) / TOTAL_PHASE

## 5. Check Bonus: 50 → 250

Current CHECK_BONUS = 50 is too weak. Increased to 250.

When Black is in check: score += 250
When Red is in check: score -= 250

## 6. Initiative Bonus (New)

Track in search `ThreadContext`:
- `last_move_aggressive: bool` — set to true if last move was capture, check, or attack

Evaluation: if `last_move_aggressive` && current_side == last_mover_side, +20 bonus.

Note: this is a simplification — the full initiative tracking would use move history, but this captures the basic idea without complex state.

## 7. Pawn Advancements

### Back Rank Threat: +80
Pawn on enemy's back rank (Red: y=0, Black: y=9) and can advance to promotion: +80

### Crossed River (no further progress): +5
Pawn that has crossed (Red: y≤4, Black: y≥5) but not on back rank: +5

### Central File Advancement: +10 per rank
Pawn on file 4 or 5 (central files), bonus scales with how many ranks it has advanced past the river.

## 8. Piece Activity

### Chariot on Enemy Back Two Rows: +40
Chariot on y ∈ [0,1] (Red's perspective) or y ∈ [8,9] (Black's perspective): +40

### Horse with Forward Progress: +20
Horse on ranks 5-6 (Red's side) or 3-4 (Black's side) with forward momentum: +20

## 9. Elephant Pair (Enhanced)

### Mutual Protection Bonus
When both elephants exist and can defend each other (each can move to a square the other protects): +60 in midgame, +40 in endgame.

This replaces the previous simpler pair bonus.

## 10. Summary of Value Changes

| Component | Old Value | New Value |
|-----------|-----------|-----------|
| CHECK_BONUS | 50 | 250 |
| Chariot mobility | ×1 | ×5 |
| King attack bonus | 0 | +25 per piece |
| Attack pressure (near enemy king) | N/A | +15 chariot/cannon, +10 horse, +5 pawn |
| Hanging piece bonus | N/A | +30 |
| Initiative bonus | N/A | +20 after aggressive move |
| Pawn promotion threat | 0 | +80 |
| Pawn crossed river | 0 | +5 |
| Chariot on enemy back rank | 0 | +40 |
| Horse forward progress | 0 | +20 |
| Elephant pair mutual protection | 0 | +60 mg / +40 eg |

## Implementation Notes

All changes are in `src/main.rs` in the `eval` module.

The main `evaluate()` function (line ~2909) will be modified to call the new `attack_rewards()` function and use increased weights throughout.

Search module's `ThreadContext` will gain a `last_move_aggressive: bool` field for initiative tracking.