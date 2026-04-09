# Evaluation Rewrite Design

**Date:** 2026-04-09
**Project:** better_rust_chinese_chess

## Status
- [ ] User approved design
- [ ] Implementation plan written
- [ ] Code implemented
- [ ] Tests passing

---

## 1. Overview

Rewrite the position evaluation function with a clean, traditional piece-square table architecture. The current evaluation has architectural problems: O(90×90) attack checks, inconsistent scales, and missing important Xiangqi concepts.

## 2. Current Problems

1. **center_control**: O(90×90) attack checking with expensive `generate_*_moves()` calls
2. **piece_coordination**: Complex heuristics that don't reflect Xiangqi strategy
3. **Mobility functions**: Called for every piece but semantics unclear
4. **Inconsistent scales**: king_safety ±200, center_control ±100, etc.
5. **Missing concepts**: Doubled pawn penalty, passed pawn evaluation, elephant river crossing

## 3. New Architecture

```
evaluate(board, side) -> i32
├── Material
│   ├── Piece values (MG/EG interpolated)
│   └── Penalties (doubled pawns, back-rank pawns)
├── PST (Piece-Square Tables)
│   ├── King PST
│   ├── Advisor PST
│   ├── Elephant PST
│   ├── Horse PST
│   ├── Cannon PST
│   ├── Chariot PST
│   └── Pawn PST
├── PositionTerms
│   ├── Mobility (Horse: jumps available, Chariot: open lines)
│   └── CenterControl (simplified attack counting)
├── KingSafety (only when king threatened)
└── PawnBonus
    ├── Linked pawns (adjacent files)
    └── Passed pawns (flank advancement)
```

## 4. Material Evaluation

### Piece Values

| Piece    | MG Value | EG Value |
|----------|----------|----------|
| King     | 10000    | 10000   |
| Chariot  | 650      | 700     |
| Horse    | 350      | 280     |
| Cannon   | 380      | 380     |
| Advisor  | 135      | 140     |
| Elephant | 105      | 100     |
| Pawn     | 80       | 200     |

### Penalties

- **Doubled pawn**: -30 (same x-coordinate, any y, same color)
- **Back-rank pawn**: -20 (y=6 for Red, y=3 for Black - not crossed river)

## 5. Piece-Square Tables

### King PST (MG/EG)

Palace corners are safest. Edge protection bonus.

```
MG:  y=0: [0,0,0,50,30,50,0,0,0]  // corners protected by edge
     y=1: [0,0,0,30,20,30,0,0,0]  // side centers
     y=2: [0,0,0,40,25,40,0,0,0]  // front corners

EG:  y=0: [0,0,0,40,25,40,0,0,0]  // king can be slightly more active in EG
     y=1: [0,0,0,25,15,25,0,0,0]
     y=2: [0,0,0,30,20,30,0,0,0]
```

### Advisor PST

Diagonal positions flanking the king.

```
MG/EG:
     y=0: [0,0,0,30,0,30,0,0,0]
     y=1: [0,0,0,0,40,0,0,0,0]   // center of palace
     y=2: [0,0,0,30,0,30,0,0,0]
```

### Elephant PST

Back-rank diagonals preferred. River positions are EXPOSED (lower value).

```
MG:
     y=0: [0,0,20,0,0,0,20,0,0]  // back-rank corners
     y=2: [10,0,30,0,20,0,30,0,10]  // near-back-rank
     y=3: [0,0,0,0,0,0,0,0,0]   // river - EXPOSED, low value
     y=4: [0,0,0,0,0,0,0,0,0]   // river - EXPOSED
```

### Horse PST

Central but avoiding knee trap. Standard horse centrality pattern.

```
MG:
     y=0: [5,10,20,20,20,20,20,10,5]
     y=1: [5,15,30,40,50,40,30,15,5]
     y=2: [10,30,50,70,80,70,50,30,10]
     y=3: [20,40,60,80,90,80,60,40,20]  // center
```

### Cannon PST

Platform control important. Center and flank positions valuable.

```
MG:
     y=0: [0,0,0,0,0,0,0,0,0]
     y=1: [20,30,40,50,60,50,40,30,20]  // behind pawns
     y=2: [30,40,50,60,70,60,50,40,30]
     y=3: [40,50,60,80,90,80,60,50,40]
```

### Chariot PST

Strong rook-like centrality. Open lines critical.

```
MG:
     y=0: [10,20,30,40,50,40,30,20,10]
     y=1: [20,30,40,50,60,50,40,30,20]
     y=2: [30,40,50,60,70,60,50,40,30]
     y=3: [40,50,60,80,90,80,60,50,40]
```

### Pawn PST

Linear progression from back to front. High value after crossing river.

```
MG:
     y=0: [0,0,0,0,0,0,0,0,0]     // enemy back rank
     y=1: [0,0,0,0,0,0,0,0,0]
     y=2: [0,0,0,0,0,0,0,0,0]
     y=3: [80,100,120,140,160,140,120,100,80]  // crossed river
     y=4: [60,80,100,120,140,120,100,80,60]   // near river
     y=5: [40,50,60,70,80,70,60,50,40]
     y=6: [20,30,40,50,60,50,40,30,20]   // starting position
     y=7: [10,20,30,40,50,40,30,20,10]
     y=8: [0,0,0,0,0,0,0,0,0]
     y=9: [0,0,0,0,0,0,0,0,0]

EG:  Passed pawns are VERY valuable
     y=0: [300,350,400,450,500,450,400,350,300]
```

## 6. Position Terms

### Mobility

Only for Horse and Chariot (the mobile pieces):

**Horse mobility**: Number of valid jumps × 10
**Chariot mobility**: Empty squares in open lines × 5

### Center Control

Simplified attack counting:
- Core area: x=3-5, y=3-6
- Weight: chariot/cannon=2, horse/pawn=1
- Net attacks × 5

## 7. King Safety

Only computed when king is threatened:

**Threats within 3 squares:**
| Piece | Weight |
|-------|--------|
| Chariot | 100 |
| Cannon | 80 |
| Horse | 60 |
| Pawn | 40 |

**Shields (same color pieces adjacent to king in palace):**
| Piece | Bonus |
|-------|-------|
| Advisor | 30 |
| Elephant | 20 |
| Other | 10 |

## 8. Pawn Bonus

**Linked pawns** (horizontally adjacent same-color pawns): +15 each
**Passed pawns** (no enemy pawns ahead on file): +50 if crossed river

## 9. Phase Interpolation

```
phase = sum(PHASE_WEIGHTS) / TOTAL_PHASE
score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
```

Where PHASE_WEIGHTS = [0, 1, 1, 1, 4, 4, 8] for [King, Advisor, Elephant, Pawn, Horse, Cannon, Chariot]

## 10. Implementation Notes

1. Single pass through board for material + PST (combine loops)
2. Separate passes only for terms that genuinely need it
3. All values are i32, no floating point
4. Clear separation: evaluate() calls sub-functions, each computes one term
5. Tests verify: material counts, PST lookup, king safety on check positions
