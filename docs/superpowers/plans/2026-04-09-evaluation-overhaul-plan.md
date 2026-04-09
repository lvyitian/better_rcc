# Evaluation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine play aggressively by increasing offensive evaluation signals — attack rewards, check pressure, mobility, pawn advancement, and initiative.

**Architecture:** All evaluation changes live in `src/main.rs` in the `eval` module (lines ~2148-2986) and search module's `ThreadContext`. Changes are additive and weight-adjusting — no architecture restructuring.

**Tech Stack:** Rust, no external dependencies

---

## Task 1: CHECK_BONUS: 50 → 250

**Files:**
- Modify: `src/main.rs:163`

- [ ] **Step 1: Change CHECK_BONUS constant**

```rust
// Old (line 163):
pub const CHECK_BONUS: i32 = 50;

// New:
pub const CHECK_BONUS: i32 = 250;
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): increase check bonus from 50 to 250 for aggressive play"
```

---

## Task 2: Mobility Normalization — Chariot ×1 → ×5

**Files:**
- Modify: `src/main.rs:2946-2948`

- [ ] **Step 1: Find and update chariot mobility multiplier**

The current code at line ~2946-2948:
```rust
PieceType::Chariot => {
    let mob = chariot_mobility(board, pos, piece.color);
    score += mob * sign;  // was *1
}
```

Change to:
```rust
PieceType::Chariot => {
    let mob = chariot_mobility(board, pos, piece.color);
    score += mob * sign * 5;  // now *5
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): normalize chariot mobility to ×5"
```

---

## Task 3: Add King Attack Bonus (+25 per piece)

**Files:**
- Modify: `src/main.rs:2960-2965` (inside `evaluate()` after king_safety calls)

- [ ] **Step 1: Read current king_safety block**

At lines ~2960-2965:
```rust
if let Some(ks_red) = king_safety(board, Color::Red, phase) {
    score += ks_red;
}
if let Some(ks_black) = king_safety(board, Color::Black, phase) {
    score += ks_black;
}
```

Add king attack bonus after this block. Insert before line 2967 (pawn_structure):

```rust
// King attack bonus: pieces directly threatening enemy king
if let Some((rk, bk)) = {
    let (r, b) = board.find_kings();
    Some((r, b))
} {
    for (king_color, enemy_king_pos) in [(Color::Red, bk), (Color::Black, rk)] {
        if let Some(kp) = enemy_king_pos {
            let attackers = board.attacks_on_square(kp);
            let our_color = king_color; // attackers are of enemy color
            let attack_count = attackers
                .iter()
                .filter(|(c, _)| *c == our_color)
                .count() as i32;
            let bonus = attack_count * 25 * king_color.sign();
            score += bonus;
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add king attack bonus (+25 per piece threatening enemy king)"
```

---

## Task 4: New `attack_rewards()` Function

**Files:**
- Modify: `src/main.rs` — add new function after `count_dir_attacks()` (around line 2595)
- Modify: `src/main.rs:2967` — call attack_rewards in evaluate()

- [ ] **Step 1: Add attack_rewards function after count_dir_attacks() (after line 2595)**

```rust
/// Evaluate attack rewards: hanging pieces, overload, central attacks, king attacks
/// Motivates aggressive play by rewarding actual threats
fn attack_rewards(board: &Board, color: Color, phase: i32) -> i32 {
    let mut score = 0;
    let mg_factor = phase;
    let eg_factor = TOTAL_PHASE - phase;

    let enemy = color.opponent();

    // Collect all our attacks and their targets
    let mut our_attacks: Vec<Coord> = Vec::new();
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(p) = board.get(pos) && p.color == color {
                let attacks = board.attacks_on_square(pos); // use board attack generation
                for (_, target) in attacks {
                    our_attacks.push(target);
                }
            }
        }
    }

    // 4.4 King attack bonus: pieces directly threatening enemy king
    let (rk, bk) = board.find_kings();
    let enemy_king_pos = if color == Color::Red { bk } else { rk };
    if let Some(ek) = enemy_king_pos {
        let king_attacks: usize = our_attacks.iter().filter(|&&a| a == ek).count();
        let king_bonus = (king_attacks as i32) * (25 * mg_factor + 15 * eg_factor) / TOTAL_PHASE;
        score += king_bonus * color.sign();
    }

    // 4.3 Central attack bonus: pieces attacking core area (x ∈ [3,5], y ∈ [4,5])
    let central_attacks = our_attacks.iter().filter(|&&pos| {
        pos.x >= 3 && pos.x <= 5 && pos.y >= 4 && pos.y <= 5
    }).count();
    let central_bonus = (central_attacks as i32) * (10 * mg_factor + 5 * eg_factor) / TOTAL_PHASE;
    score += central_bonus * color.sign();

    // 4.1 Hanging piece bonus: enemy piece with 0 or 1 defender and no higher-value protector
    for y in 0..10 {
        for x in 0..9 {
            let pos = Coord::new(x as i8, y as i8);
            if let Some(p) = board.get(pos) && p.color == enemy {
                // Count our attackers on this piece
                let attacked_by_us = our_attacks.iter().filter(|&&a| a == pos).count();

                // Count their defenders
                let mut defended_by = 0;
                for dy in 0..10 {
                    for dx in 0..9 {
                        let def_pos = Coord::new(dx as i8, dy as i8);
                        if let Some(dp) = board.get(def_pos)
                            && dp.color == enemy
                            && dp.piece_type != p.piece_type
                        {
                            // Check if this piece actually defends pos
                            let def_attacks = board.attacks_on_square(def_pos);
                            if def_attacks.iter().any(|(c, target)| c == &enemy && target == pos) {
                                defended_by += 1;
                            }
                        }
                    }
                }

                // Hanging: no defender or only 1 and we're attacking with multiple
                if attacked_by_us > 0 {
                    if defended_by == 0 || (defended_by == 1 && attacked_by_us > 1) {
                        let hanging_value = (30 * mg_factor + 20 * eg_factor) / TOTAL_PHASE;
                        score += hanging_value * color.sign();
                    }
                }
            }
        }
    }

    score * color.sign()
}
```

- [ ] **Step 2: Add call to attack_rewards in evaluate() (around line 2967)**

After line:
```rust
score += elephant_structure(board, Color::Red, phase);
score += elephant_structure(board, Color::Black, phase);
```

Add:
```rust
// Attack rewards
score += attack_rewards(board, Color::Red, phase);
score += attack_rewards(board, Color::Black, phase);
```

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add attack rewards function (hanging pieces, central attacks, king attacks)"
```

---

## Task 5: Pawn Advancement Rewards

**Files:**
- Modify: `src/main.rs:2785-2843` (`pawn_structure` function)

- [ ] **Step 1: Read the current pawn_structure function (lines ~2785-2843)**

Current implementation scans for pawns and evaluates:
- Doubled pawn penalty (-30)
- Back-rank penalty (-20)
- Linked pawn bonus (+20-30)

Add advancement rewards inside the same loop after linked pawn bonus (around line 2827):

```rust
// Advancement bonus: reward pushing pawns forward
let crossed = pos.crosses_river(color);
let advancement_bonus = if crossed {
    // Crossed river — check for back rank threat or further progress
    match color {
        Color::Red => {
            if pos.y == 0 { 80 }  // Red pawn on Black's back rank (threat to promote)
            else if pos.y <= 4 { 5 }  // Crossed but not further advanced
            else { 0 }
        },
        Color::Black => {
            if pos.y == 9 { 80 }  // Black pawn on Red's back rank
            else if pos.y >= 5 { 5 }  // Crossed but not further advanced
            else { 0 }
        }
    }
} else {
    0
};
score += advancement_bonus;

// Central file bonus: pawn on file 4 or 5 (columns 4-5) gets bonus per rank advanced
if pos.x == 4 || pos.x == 5 {
    let ranks_from_origin = match color {
        Color::Red => 6 - pos.y,  // Red starts at y=6, higher = more advanced
        Color::Black => pos.y - 3,  // Black starts at y=3
    };
    if ranks_from_origin > 0 {
        score += ranks_from_origin * 10 * color.sign();
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add pawn advancement rewards (promotion threat +80, crossed river +5, central files +10)"
```

---

## Task 6: Elephant Pair Mutual Protection Bonus

**Files:**
- Modify: `src/main.rs:2885-2902` (in `elephant_structure` — the "2 elephants" case)

- [ ] **Step 1: Enhance the 2-elephant case in elephant_structure (around line 2885)**

Current code checks if they share moves (linked). Replace with enhanced mutual protection:

```rust
2 => {
    let pos1 = elephants[0];
    let pos2 = elephants[1];

    // Generate moves for both elephants
    let moves1 = movegen::generate_elephant_moves(board, pos1, color);
    let moves2 = movegen::generate_elephant_moves(board, pos2, color);

    // Check mutual protection: each elephant can move to a square the other protects
    let mut mutual_protection = false;

    // Elephant 1's moves: these are the squares elephant 1 can reach
    // Check if elephant 2's position is in elephant 1's reach AND vice versa
    // Or more precisely: they protect each other if they can move to defend positions
    for m1 in &moves1 {
        // Does elephant 2 also have this move? (they occupy shared defensive positions)
        if moves2.contains(m1) {
            mutual_protection = true;
            break;
        }
    }

    if mutual_protection {
        score += (60 * mg_factor + 40 * eg_factor) / TOTAL_PHASE;
    } else {
        // Even without direct mutual protection, having 2 elephants is still a structure bonus
        score += (30 * mg_factor + 20 * eg_factor) / TOTAL_PHASE;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): enhance elephant pair bonus with mutual protection (+60 mg/+40 eg)"
```

---

## Task 7: Chariot & Horse Activity Bonuses

**Files:**
- Modify: `src/main.rs:2935-2955` (inside evaluate() loop for each piece)

- [ ] **Step 1: Add activity bonuses for Chariot and Horse inside the piece loop**

After the existing mobility match arms, add:

```rust
// Chariot activity: on enemy back two rows
match piece.piece_type {
    PieceType::Chariot => {
        let mob = chariot_mobility(board, pos, piece.color);
        score += mob * sign * 5;
        // Activity bonus: on enemy back two rows
        let on_enemy_back_rank = match piece.color {
            Color::Red => pos.y <= 1,    // Red attacking Black's back rank (y=8,9)
            Color::Black => pos.y >= 8,  // Black attacking Red's back rank (y=0,1)
        };
        if on_enemy_back_rank {
            score += 40 * sign;
        }
    }
    PieceType::Horse => {
        let mob = horse_mobility(board, pos, piece.color);
        score += mob * sign * 5;
        // Activity bonus: forward progress (5th/6th rank)
        let forward_rank = match piece.color {
            Color::Red => pos.y >= 3 && pos.y <= 5,  // Red's forward positions
            Color::Black => pos.y >= 4 && pos.y <= 6, // Black's forward positions
        };
        if forward_rank {
            score += 20 * sign;
        }
    }
    // ... cannon etc unchanged
}
```

Note: Also update Horse mobility from ×5 to ×5 (it already is ×5) — check and confirm.

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add chariot/horse activity bonuses (chariot on enemy back rank +40, horse forward +20)"
```

---

## Task 8: King Safety — Attack Pressure Near Enemy King

**Files:**
- Modify: `src/main.rs:2720-2762` (`king_safety` function)

- [ ] **Step 1: Read current king_safety (lines ~2720-2762)**

Add attack pressure scoring inside the existing function, after the PALACE_DELTAS defensive bonus block (around line 2739):

```rust
// Inside king_safety, after:
// for (dx, dy) in PALACE_DELTAS { ... }
// Add:

// 2. Attack pressure near enemy king
let enemy_king_pos = match color {
    Color::Red => bk,
    Color::Black => rk,
};

if let Some(ek) = enemy_king_pos {
    // Check adjacent squares to enemy king
    let adj_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  // orthogonal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  // diagonal
    ];

    for (dx, dy) in adj_offsets {
        let adj_pos = Coord::new(ek.x + dx, ek.y + dy);
        if adj_pos.is_valid() {
            if let Some(p) = board.get(adj_pos) && p.color == color {
                let pressure = match p.piece_type {
                    PieceType::Chariot | PieceType::Cannon => 15,
                    PieceType::Horse => 10,
                    PieceType::Pawn if p.piece_type.to_string().contains("crossed_river") || adj_pos.crosses_river(color) => 5,
                    _ => 0,
                };
                safety += pressure;
            }
        }
    }
}
```

Wait — the pawn check needs the actual piece's position, not adj_pos. Better approach:

```rust
// Attack pressure near enemy king — scan all our pieces
for y in 0..10 {
    for x in 0..9 {
        let our_pos = Coord::new(x as i8, y as i8);
        if let Some(p) = board.get(our_pos) && p.color == color {
            // Can this piece attack adjacent squares of enemy king?
            let dist_to_king = our_pos.distance_to(ek);
            if dist_to_king <= 2 {  // Within 2 squares of enemy king
                let pressure = match p.piece_type {
                    PieceType::Chariot | PieceType::Cannon => 15,
                    PieceType::Horse => 10,
                    PieceType::Pawn if our_pos.crosses_river(color) => 5,
                    _ => 0,
                };
                safety += pressure;
            }
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add attack pressure scoring in king safety (pieces near enemy king get bonus)"
```

---

## Task 9: Initiative Bonus in ThreadContext

**Files:**
- Modify: `src/main.rs:3084-3099` (`ThreadContext` struct)

- [ ] **Step 1: Add last_move_aggressive to ThreadContext**

Around line 3084-3088, the struct is:
```rust
pub struct ThreadContext {
    pub history_table: [[i32; 90]; 90],
    pub killer_moves: [[Option<Action>; 2]; (MAX_DEPTH + 4) as usize],
    pub counter_moves: [[Option<Action>; 90]; 90],
}
```

Add:
```rust
pub struct ThreadContext {
    pub history_table: [[i32; 90]; 90],
    pub killer_moves: [[Option<Action>; 2]; (MAX_DEPTH + 4) as usize],
    pub counter_moves: [[Option<Action>; 90]; 90],
    pub last_move_aggressive: bool,  // NEW: for initiative bonus
}
```

- [ ] **Step 2: Initialize in Default and new()**

In `impl Default for ThreadContext` and `impl ThreadContext`, initialize `last_move_aggressive: false`.

- [ ] **Step 3: Update in search when making aggressive moves**

In the search function, when a move is a capture, check, or attack, set `last_move_aggressive = true`. After a quiet move, set to false.

- [ ] **Step 4: Apply initiative bonus in evaluate()**

In evaluate(), call initiative bonus. Since `ThreadContext` is passed through search but not into eval directly, we need to pass the flag through. Simplest approach: add a parameter to `evaluate()`:

```rust
pub fn evaluate(board: &Board, side: Color, initiative: bool) -> i32 {
    // ...
    if initiative {
        score += 20 * side.sign();
    }
    // ...
}
```

Or use a thread-local: `thread_local::thread_local!` for simplicity, but the parameter approach is cleaner.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): add initiative bonus tracking in ThreadContext"
```

---

## Task 10: Integration — Call all new functions in evaluate()

**Files:**
- Modify: `src/main.rs:2909-2985` (the main evaluate function)

- [ ] **Step 1: Read the full evaluate function and ensure all new components are called**

Current order (lines ~2909-2985):
1. Endgame tablebase probe
2. King existence check
3. phase calculation
4. Per-piece loop: material+PST + mobility
5. king_safety (Red + Black)
6. pawn_structure (Red + Black)
7. center_control (Red + Black)
8. piece_coordination (Red + Black)
9. elephant_structure (Red + Black)
10. check_bonus
11. color.sign() return

Add in order:
- After elephant_structure (line ~2976): attack_rewards (Red + Black)
- Before check_bonus (line ~2978): king attack bonus (already done in Task 3)
- Initiative bonus: passed as parameter or via ThreadContext

- [ ] **Step 2: Run tests**

```bash
cargo test --lib
```

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(eval): integrate all new evaluation components"
```

---

## Summary of Changes

| Task | Constant/Function | Change |
|------|-------------------|--------|
| 1 | CHECK_BONUS | 50 → 250 |
| 2 | chariot_mobility in evaluate | ×1 → ×5 |
| 3 | evaluate() | + king attack bonus |
| 4 | attack_rewards() | NEW function |
| 5 | pawn_structure() | + advancement rewards |
| 6 | elephant_structure() | + mutual protection bonus |
| 7 | evaluate() piece loop | + chariot/horse activity |
| 8 | king_safety() | + attack pressure |
| 9 | ThreadContext | + initiative tracking |
| 10 | evaluate() | integrate all new components |

**Total: 10 tasks, ~10 commits**