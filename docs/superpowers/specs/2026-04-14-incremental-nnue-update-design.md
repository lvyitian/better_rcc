# Incremental NNUE Update — Design

## Status

Approved 2026-04-14.

## Motivation

The current NNUE evaluation (`nn_evaluate_or_handcrafted`) recomputes the 1260-dimensional input features from scratch on every call, iterating all 90 squares. During search, the same position is often evaluated thousands of times across the search tree. The classic NNUE technique (Stockfish's approach) caches the 1024-dimensional feature-transform accumulators per position and updates them incrementally with O(1) work per move, reducing evaluation cost by ~99% in the search tree.

## Design Goals

1. Incremental updates on `make_move` / `undo_move` for search speed
2. Dirty-flag recomputation for first evaluation at search leaves
3. Global transposition cache for positions reached without `make_move` (transpositions)
4. Correctness: incremental evaluation must be bit-exact to full recomputation
5. No changes to the `NNUEFeedForward` public API

---

## 1. Storage Architecture

### 1.1 `NnueState` struct

Added to the `Board` struct as field `pub nnue_state: NnueState`:

```rust
/// Persists across make/undo. Reset on position setup (new board, FEN load).
pub struct NnueState {
    /// Feature-transform accumulator from Red's perspective.
    pub red_acc:   Accumulator,
    /// Feature-transform accumulator from Black's perspective.
    pub black_acc: Accumulator,
    /// Non-king piece count from Red's perspective.
    pub red_nonc:  u8,
    /// Non-king piece count from Black's perspective.
    pub black_nonc: u8,
    /// When true, accumulators are stale; next evaluation must recompute or cache-hit.
    pub dirty:     bool,
}
```

`Accumulator` is the existing `#[repr(C)] pub struct Accumulator { pub vals: [i16; 1024] }` from `nn_eval.rs`.

The dirty flag is a single `bool`. A single `make_move` always dirtifies; `undo_move` restores a clean prior snapshot.

### 1.2 Global Transposition Cache

In `nn_eval.rs`:

```rust
use std::sync::RwLock;
use std::collections::HashMap;

type Accumulators = (Accumulator, Accumulator, u8, u8); // red_acc, black_acc, red_nonc, black_nonc

static NNUE_CACHE: RwLock<HashMap<u64, Accumulators>> = RwLock::new(HashMap::new());
const MAX_CACHE_ENTRIES: usize = 65536;
```

- **Key**: `board.zobrist_key` (existing 64-bit Zobrist of position)
- **Value**: Both perspective accumulators and non-king counts
- **Eviction**: When `len() > MAX_CACHE_ENTRIES`, remove the oldest 25% of entries (simple strategy; Xiangqi trees are shallow)
- **Thread safety**: `RwLock` — readers proceed concurrently; writers are exclusive

The global cache stores the **accumulator state** of a position. On `make_move`, before dirtifying, the current position's accumulators are inserted into the cache (since that position may be reached again via a different move order).

---

## 2. Update on `make_move`

### 2.1 Feature index computation

The NNUE feature index for a piece is:

```
feature_index(color, piece_type, sq) =
    if color == Red:  piece_type * 90 + sq   (base 0 block)
    else:              630 + piece_type * 90 + sq   (base 630 block)
```

where `sq = y * 9 + x` (0 ≤ sq ≤ 89).

The NTM vertical flip (`y → 9 - y`) is already baked into the initial encoding of the NTM block. During incremental updates, we simply add/remove feature indices as pieces appear on physical squares — no coordinate transform needed.

### 2.2 Per-perspective update rule

When a move is made by color `c` from square `src_sq` to `dst_sq`, for each perspective `p` ∈ {Red, Black}:

- **Own piece block (base 0)**: If `c == p`, remove `ft_weights[old_idx]` and add `ft_weights[new_idx]`.
- **Opponent piece block (base 630)**: If `c != p`, remove `ft_weights[old_idx]` and add `ft_weights[new_idx]`.

If `c == p` and there is a capture, also remove the captured piece from the opponent's block. If `c != p` and there is a capture, also remove the captured piece from the own block.

### 2.3 Incremental non-king count update

`red_nonc` / `black_nonc` are updated directly:
- Moving a non-king piece: no net change
- Capturing a non-king piece: decrement by 1
- Capturing a king: no change to non-king count (kings are excluded from the count)

### 2.4 Algorithm

```
apply_incremental_update(board, src_sq, dst_sq, moved_piece, captured):
    for perspective in [Red, Black]:
        // Determine which block the moved piece belongs to in this perspective
        moved_block_base = if moved_piece.color == perspective { 0 } else { 630 }
        moved_old_idx = moved_block_base + moved_piece.piece_type * 90 + src_sq
        moved_new_idx = moved_block_base + moved_piece.piece_type * 90 + dst_sq

        // Update accumulator for this perspective
        for i in 0..FT_DIM:
            persp_acc = if perspective == Red { &mut board.nnue_state.red_acc }
                        else { &mut board.nnue_state.black_acc };
            persp_acc.vals[i] -= nn.ft_weights[moved_old_idx].vals[i];
            persp_acc.vals[i]  = persp_acc.vals[i].saturating_add(nn.ft_weights[moved_new_idx].vals[i]);

        // Handle captured piece
        if let Some(cp) = captured:
            // Captured piece belongs to the OPPONENT of the mover
            capt_block_base = if cp.color == Red { 0 } else { 630 }
            capt_idx = capt_block_base + cp.piece_type * 90 + dst_sq

            // From the perspective's view:
            // If cp.color == perspective → it's in the opponent block (base 630)
            // If cp.color != perspective → it's in the own block (base 0)
            // But this is always the dst_sq because captured piece was on dst before move
            for i in 0..FT_DIM:
                persp_acc.vals[i] -= nn.ft_weights[capt_idx].vals[i];

            // Update non-king count
            if cp.piece_type != PieceType::King:
                if perspective == Red:  board.nnue_state.red_nonc  -= 1;
                else:                  board.nnue_state.black_nonc -= 1;

    board.nnue_state.dirty = true;
```

Note: the captured piece is on `dst_sq` — its index in both perspectives is determined by its `color`, not by whose move it was.

### 2.5 Cache insertion before dirtify

Before setting `dirty = true`, insert the current (pre-move) accumulator state into the global cache:

```rust
let key = board.zobrist_key;
let val = (board.nnue_state.red_acc.clone(), board.nnue_state.black_acc.clone(),
           board.nnue_state.red_nonc, board.nnue_state.black_nonc);
insert_into_cache(key, val);
```

---

## 3. After `make_move` — Dirty State

After the incremental update and cache insertion:

```rust
board.nnue_state.dirty = true;
```

The next call to `nn_evaluate_or_handcrafted` must check `dirty` before using cached accumulators.

---

## 4. After `undo_move`

Undo does NOT apply incremental updates in reverse. Instead, it restores the `NnueState` from the snapshot saved before `make_move`.

A minimal snapshot struct is stored in the `Action` history entry (or a parallel history):

```rust
pub struct NnueSnapshot {
    pub red_acc:   Accumulator,
    pub black_acc:  Accumulator,
    pub red_nonc:  u8,
    pub black_nonc: u8,
}
```

On `make_move`, save snapshot to `Action.nnue_snapshot`. On `undo_move`, restore from `Action.nnue_snapshot` and remove the corresponding entry from the global cache (pop by key).

---

## 5. First Evaluation Path

Modified `nn_evaluate_or_handcrafted(board, side, initiative)`:

```rust
pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    let handcrafted = handcrafted_evaluate(board, side, initiative);

    let (acc, nonc) = if !board.nnue_state.dirty {
        // Clean: use stored accumulators directly
        if side == Color::Red {
            (&board.nnue_state.red_acc, board.nnue_state.red_nonc)
        } else {
            (&board.nnue_state.black_acc, board.nnue_state.black_nonc)
        }
    } else {
        // Dirty: check global cache
        if let Some((ra, ba, rn, bn)) = nnue_cache_get(board.zobrist_key) {
            board.nnue_state.red_acc   = ra;
            board.nnue_state.black_acc  = ba;
            board.nnue_state.red_nonc  = rn;
            board.nnue_state.black_nonc = bn;
            board.nnue_state.dirty = false;
            if side == Color::Red {
                (&board.nnue_state.red_acc, board.nnue_state.red_nonc)
            } else {
                (&board.nnue_state.black_acc, board.nnue_state.black_nonc)
            }
        } else {
            // Recompute from scratch
            let (stm, ntm) = NNInputPlanes::from_board(board);
            let (ra, ba) = NN_NET.compute_accumulators(&stm.data, &ntm.data);
            let rn = crate::nnue_input::count_non_king_pieces_from(board, Color::Red);
            let bn = crate::nnue_input::count_non_king_pieces_from(board, Color::Black);
            board.nnue_state.red_acc   = ra;
            board.nnue_state.black_acc  = ba;
            board.nnue_state.red_nonc  = rn;
            board.nnue_state.black_nonc = bn;
            board.nnue_state.dirty = false;
            // Store in cache
            nnue_cache_insert(board.zobrist_key, (ra, ba, rn, bn));
            if side == Color::Red {
                (&board.nnue_state.red_acc, board.nnue_state.red_nonc)
            } else {
                (&board.nnue_state.black_acc, board.nnue_state.black_nonc)
            }
        }
    };

    let output = NN_NET.forward_from_accumulators(acc, nonc);
    // ... blending with handcrafted (unchanged)
}
```

The new private method `forward_from_accumulators(&self, acc: &Accumulator, non_king_count: u8) -> NNOutput` runs the SCReLU + dot-product step without recomputing accumulators. The existing `forward()` method is refactored to call this internally.

---

## 6. `forward_from_accumulators` Helper

Extract from the existing `forward()` the post-accumulator computation:

```rust
/// Internal helper: applies SCReLU and output layer to pre-computed accumulators.
/// `stm_acc` is the feature-transform accumulator for the side to move,
/// `ntm_acc` for the opponent.
pub fn forward_from_accumulators(
    &self,
    stm_acc: &Accumulator,
    ntm_acc: &Accumulator,
    non_king_count: u8,
) -> NNOutput {
    let bucket_idx = bucket_index(non_king_count);

    // SCReLU: clamp(x, 0, QA), then square, returning i32
    let mut combined = [0i32; FT_DIM * 2];
    for i in 0..FT_DIM {
        combined[i]     = screlu(stm_acc.vals[i]);
        combined[FT_DIM + i] = screlu(ntm_acc.vals[i]);
    }

    // Dot product with bucket's output weights
    let mut raw: i64 = 0;
    for i in 0..FT_DIM * 2 {
        raw += i64::from(self.out_weights[bucket_idx][i]) * i64::from(combined[i]);
    }

    // Normalization
    let qa_f = QA as f32;
    let qb_f = QB as f32;
    let scale_f = SCALE as f32;
    let raw_f = raw as f32 / (qa_f * qb_f * qa_f);
    let raw_result = ((raw_f / qa_f) + f32::from(self.out_bias[bucket_idx])) * scale_f / (qa_f * qb_f);
    let nn_score = raw_result.tanh() * scale_f;

    // Extract per-bucket values for correction term
    let mut bucket_vals = [0f32; NUM_BUCKETS];
    for b in 0..NUM_BUCKETS {
        let mut raw_b: i64 = 0;
        for i in 0..FT_DIM * 2 {
            raw_b += i64::from(self.out_weights[b][i]) * i64::from(combined[i]);
        }
        let raw_b_f = raw_b as f32 / (qa_f * qb_f * qa_f);
        bucket_vals[b] = (raw_b_f / qa_f + f32::from(self.out_bias[b])) * scale_f / (qa_f * qb_f);
    }
    let avg_bucket_val: f32 = bucket_vals.iter().sum::<f32>() / NUM_BUCKETS as f32;
    let correction = bucket_vals[bucket_idx] - avg_bucket_val;

    NNOutput { nn_score, correction }
}
```

The existing `forward()` becomes:

```rust
pub fn forward(&self, stm: &[f32; INPUT_DIM], ntm: &[f32; INPUT_DIM], non_king_count: u8) -> f32 {
    let (stm_acc, ntm_acc) = self.compute_accumulators(stm, ntm);
    self.forward_from_accumulators(&stm_acc, &ntm_acc, non_king_count).nn_score
}
```

---

## 7. Testing

### 7.1 Correctness — make/undo cycle

After every `make_move` + `undo_move` pair, the stored accumulators must equal those recomputed from scratch:

```rust
#[test]
fn test_incremental_make_undo_cycle() {
    let mut board = Board::new(RuleSet::Official, 1);
    nnue_init_fresh_board(&mut board); // compute and store initial accumulators

    let moves: Vec<Action> = generate_all_legal_moves(&board);
    for mv in moves {
        let saved_state = board.nnue_state.clone();
        let saved_key = board.zobrist_key;

        board.make_move(mv);
        // At dirty position, evaluate fully
        let eval_dirty = nn_evaluate_or_handcrafted(&board, board.current_side, false);
        let eval_recomputed = nn_evaluate_from_scratch(&board);

        board.undo_move(mv);

        // After undo, accumulators must match saved snapshot
        assert_eq!(board.nnue_state.red_acc.vals, saved_state.red_acc.vals);
        assert_eq!(board.nnue_state.black_acc.vals, saved_state.black_acc.vals);
        assert_eq!(board.nnue_state.dirty, false);
    }
}
```

### 7.2 Equivalence — incremental vs. full recompute

At clean (non-dirty) positions, the incremental evaluation must equal the full recomputation:

```rust
#[test]
fn test_incremental_equals_recompute() {
    let mut board = Board::new(RuleSet::Official, 1);
    nnue_init_fresh_board(&mut board);

    for _ in 0..100 {
        let eval_incremental = nn_evaluate_or_handcrafted(&board, board.current_side, false);
        let eval_recomputed   = nn_evaluate_from_scratch(&board);
        assert!((eval_incremental - eval_recomputed).abs() < 1,
                "mismatch: {} vs {}", eval_incremental, eval_recomputed);

        let moves = generate_all_legal_moves(&board);
        if moves.is_empty() { break; }
        board.make_move(moves[rand usize() % moves.len()]);
    }
}
```

### 7.3 Search invariant

Running a depth-4 search with incremental updates must produce the same root move ordering (top-3) as without:

```rust
#[test]
fn test_search_move_ordering_preserved() {
    let mut board = Board::new(RuleSet::Official, 1);
    init_nnue_state(&mut board); // initialize with incremental state

    let moves_ref = search_root(&board, Depth::4, /* incremental = false */);
    let moves_inc = search_root(&board, Depth::4, /* incremental = true */);

    assert_eq!(&moves_ref[0..3], &moves_inc[0..3],
               "Top-3 root moves must be identical");
}
```

### 7.4 Cache hit rate

During a full depth-6 search on a mid-game position, the global cache hit rate should be > 60%:

```rust
#[test]
fn test_cache_hit_rate() {
    let mut board = Board::new(RuleSet::Official, 1);
    init_nnue_state(&mut board);
    let (hits, total) = benchmark_search(&board, Depth::6);
    let hit_rate = hits as f32 / total as f32;
    assert!(hit_rate > 0.6, "Cache hit rate {} below 60%", hit_rate);
}
```

---

## 8. Implementation Order

1. Add `NnueState` struct and field to `Board`
2. Add `forward_from_accumulators` helper and refactor `forward()`
3. Add global `NNUE_CACHE` with `RwLock`
4. Add `nnue_cache_get` / `nnue_cache_insert` helpers
5. Implement `apply_incremental_update` function
6. Modify `make_move` to call incremental update and save snapshot
7. Modify `undo_move` to restore `NnueState` from snapshot
8. Modify `nn_evaluate_or_handcrafted` with dirty-flag path and cache lookup
9. Add tests (Sections 7.1–7.4)
10. Benchmark: verify speedup on search (depth 4–6 self-play)

---

## 9. Open Questions

None — all resolved in design review.
