use crate::nn_eval::{Accumulator, NNUEFeedForward};
use crate::{Board, Color, Piece, PieceType};
use std::sync::RwLock;
use std::collections::HashMap;

pub const MAX_CACHE_ENTRIES: usize = 65536;

static NNUE_CACHE: std::sync::LazyLock<RwLock<HashMap<u64, (Accumulator, Accumulator, u8)>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

pub fn nnue_cache_get(key: u64) -> Option<(Accumulator, Accumulator, u8)> {
    NNUE_CACHE.read().ok()?.get(&key).cloned()
}

pub fn nnue_cache_insert(key: u64, val: (Accumulator, Accumulator, u8)) {
    let mut cache = match NNUE_CACHE.write() {
        Ok(c) => c,
        Err(e) => e.into_inner(),
    };
    if cache.len() >= MAX_CACHE_ENTRIES {
        let evict_count = MAX_CACHE_ENTRIES / 4;
        // Collect keys to evict first, then remove
        let keys_to_evict: Vec<u64> = cache.keys().take(evict_count).copied().collect();
        for k in keys_to_evict {
            cache.remove(&k);
        }
    }
    cache.insert(key, val);
}

pub fn nnue_cache_remove(key: &u64) {
    if let Ok(mut cache) = NNUE_CACHE.write() {
        cache.remove(key);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnueSnapshot {
    pub red_acc: Accumulator,
    pub black_acc: Accumulator,
    pub non_king_count: u8,
}

impl NnueSnapshot {
    pub fn from_board(board: &Board) -> Self {
        Self {
            red_acc: board.nnue_state.red_acc.clone(),
            black_acc: board.nnue_state.black_acc.clone(),
            non_king_count: board.nnue_state.non_king_count,
        }
    }
}

#[derive(Clone)]
pub struct NnueState {
    pub red_acc: Accumulator,
    pub black_acc: Accumulator,
    pub non_king_count: u8,
    pub dirty: bool,
}

impl NnueState {
    pub fn fresh(board: &Board) -> Self {
        let (stm, ntm) = crate::nnue_input::NNInputPlanes::from_board(board);
        let (red_acc, black_acc) = crate::nn_eval::NN_NET.compute_accumulators(&stm.data, &ntm.data);
        let non_king_count = crate::nnue_input::count_non_king_pieces(board);
        Self {
            red_acc,
            black_acc,
            non_king_count,
            dirty: false,
        }
    }

    pub fn zero() -> Self {
        Self {
            red_acc: Accumulator { vals: [0i16; crate::nnue_input::FT_DIM] },
            black_acc: Accumulator { vals: [0i16; crate::nnue_input::FT_DIM] },
            non_king_count: 0,
            dirty: true,
        }
    }

    /// Apply an incremental update for a move.
    /// cache_key: the zobrist key BEFORE the move (for cache insertion).
    pub fn apply_move(
        &mut self,
        src_sq: u8,
        dst_sq: u8,
        moved_piece: Piece,
        captured: Option<Piece>,
        nn: &NNUEFeedForward,
        cache_key: u64,
    ) {
        // Insert current state into global cache before dirtifying
        nnue_cache_insert(
            cache_key,
            (self.red_acc.clone(), self.black_acc.clone(), self.non_king_count),
        );

        for &perspective in &[Color::Red, Color::Black] {
            let persp_acc = if perspective == Color::Red {
                &mut self.red_acc
            } else {
                &mut self.black_acc
            };

            // Moved piece
            let moved_base = if moved_piece.color == perspective { 0 } else { 630 };
            let moved_old_idx = moved_base + moved_piece.piece_type as usize * 90 + src_sq as usize;
            let moved_new_idx = moved_base + moved_piece.piece_type as usize * 90 + dst_sq as usize;

            for i in 0..crate::nnue_input::FT_DIM {
                persp_acc.vals[i] = persp_acc.vals[i]
                    .saturating_sub(nn.ft_weights[moved_old_idx].vals[i])
                    .saturating_add(nn.ft_weights[moved_new_idx].vals[i]);
            }

            // Captured piece
            if let Some(cp) = captured {
                let capt_base = if cp.color == Color::Red { 0 } else { 630 };
                let capt_idx = capt_base + cp.piece_type as usize * 90 + dst_sq as usize;

                for i in 0..crate::nnue_input::FT_DIM {
                    persp_acc.vals[i] = persp_acc.vals[i]
                        .saturating_sub(nn.ft_weights[capt_idx].vals[i]);
                }
            }
        }

        // Decrement non_king_count ONCE, outside the perspective loop
        if let Some(cp) = captured {
            if cp.piece_type != PieceType::King {
                self.non_king_count = self.non_king_count.saturating_sub(1);
            }
        }

        self.dirty = true;
    }
}