use std::fmt;
use std::io;
use std::io::Write;
use std::sync::OnceLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::thread;

pub const BOARD_WIDTH: i8 = 9;
pub const BOARD_HEIGHT: i8 = 10;
pub const MAX_DEPTH: u8 = 14;
pub const QS_MAX_DEPTH: u8 = 8;
pub const MAX_CHECK_EXTENSION: u8 = 2;
pub const MAX_TOTAL_EXTENSION: u8 = 3;
pub const TT_SIZE: usize = 1 << 25;
pub const ENDGAME_THRESHOLD: i32 = 8000; 
pub const MIDGAME_THRESHOLD: i32 = 4000;
pub const REPETITION_VIOLATION_COUNT: u8 = 3;
pub const SEARCH_TIMEOUT_MS: u64 = 21000; 
pub const TIME_BUFFER_MS: u64 = 1000;
pub const NULL_MOVE_REDUCTION: u8 = 2;
pub const LMR_MIN_MOVES: usize = 4;
pub const SEARCH_THREADS: usize = 4;
pub const FUTILITY_MARGIN: i32 = 200;
pub const SEE_MARGIN: i32 = -50;
pub const ASPIRATION_WINDOW: i32 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleSet {
    Official,
    OnlyLongCheckIllegal,
    NoRestriction,
}

impl RuleSet {
    pub fn description(&self) -> &'static str {
        match self {
            RuleSet::Official => "正式竞赛规则：单方面长将、长捉均违规，重复3次判负",
            RuleSet::OnlyLongCheckIllegal => "宽松规则：仅长将违规，长捉不限制",
            RuleSet::NoRestriction => "娱乐规则：长将长捉均不限制，仅吃将/困毙分胜负",
        }
    }

    #[inline(always)]
    pub fn is_long_check_banned(&self) -> bool {
        match self {
            RuleSet::Official | RuleSet::OnlyLongCheckIllegal => true,
            RuleSet::NoRestriction => false,
        }
    }

    #[inline(always)]
    pub fn is_long_capture_banned(&self) -> bool {
        match self {
            RuleSet::Official => true,
            RuleSet::OnlyLongCheckIllegal | RuleSet::NoRestriction => false,
        }
    }
}

impl fmt::Display for RuleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum PieceType {
    King = 0,
    Advisor = 1,
    Elephant = 2,
    Pawn = 3,
    Horse = 4,
    Cannon = 5,
    Chariot = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    #[inline(always)]
    pub fn opponent(self) -> Self {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }

    #[inline(always)]
    pub fn sign(self) -> i32 {
        match self {
            Color::Red => 1,
            Color::Black => -1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Coord {
    pub x: i8,
    pub y: i8,
}

impl Coord {
    #[inline(always)]
    pub fn new(x: i8, y: i8) -> Self {
        Coord { x, y }
    }

    #[inline(always)]
    pub fn is_valid(self) -> bool {
        self.x >= 0 && self.x < BOARD_WIDTH && self.y >= 0 && self.y < BOARD_HEIGHT
    }

    #[inline(always)]
    pub fn in_palace(self, color: Color) -> bool {
        let x_ok = self.x >= 3 && self.x <= 5;
        let y_ok = match color {
            Color::Red => self.y >= 7,
            Color::Black => self.y <= 2,
        };
        x_ok && y_ok
    }

    #[inline(always)]
    pub fn crosses_river(self, color: Color) -> bool {
        match color {
            Color::Red => self.y <= 4,
            Color::Black => self.y >= 5,
        }
    }

    #[inline(always)]
    pub fn in_core_area(self, color: Color) -> bool {
        let x_ok = self.x >= 3 && self.x <= 5;
        let y_ok = match color {
            Color::Red => self.y >= 4 && self.y <= 7,
            Color::Black => self.y >= 2 && self.y <= 5,
        };
        x_ok && y_ok
    }

    #[inline(always)]
    pub fn distance_to(self, other: Coord) -> i32 {
        (self.x - other.x).abs() as i32 + (self.y - other.y).abs() as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    pub src: Coord,
    pub tar: Coord,
    pub captured: Option<Piece>,
    pub is_check: bool,
    pub is_capture_threat: bool,
}

impl Action {
    #[inline(always)]
    pub fn new(src: Coord, tar: Coord, captured: Option<Piece>) -> Self {
        Action {
            src,
            tar,
            captured,
            is_check: false,
            is_capture_threat: false,
        }
    }

    #[inline(always)]
    pub fn mvv_lva_score(self) -> i32 {
        const PIECE_VALUES: [i32; 7] = [10000, 100, 100, 50, 300, 300, 500];
        self.captured.map_or(0, |p| PIECE_VALUES[p.piece_type as usize]) * 100
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Zobrist {
    pub pieces: [[[u64; 7]; 2]; 90],
    pub side: u64,
}

impl Zobrist {
    fn new() -> Self {
        struct Xorshift64 {
            state: u64,
        }
        impl Xorshift64 {
            fn new(seed: u64) -> Self {
                Xorshift64 { state: seed }
            }
            fn next(&mut self) -> u64 {
                let mut x = self.state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.state = x;
                x
            }
        }

        let mut rng = Xorshift64::new(0x123456789abcdef);
        let mut pieces = [[[0; 7]; 2]; 90];
        for pos in 0..90 {
            for color in 0..2 {
                for pt in 0..7 {
                    pieces[pos][color][pt] = rng.next();
                }
            }
        }
        Zobrist {
            pieces,
            side: rng.next(),
        }
    }

    #[inline(always)]
    pub fn pos_idx(&self, coord: Coord) -> usize {
        (coord.y * BOARD_WIDTH + coord.x) as usize
    }
}

static ZOBRIST_CELL: OnceLock<Zobrist> = OnceLock::new();

pub fn get_zobrist() -> &'static Zobrist {
    ZOBRIST_CELL.get_or_init(Zobrist::new)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TTEntryType {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub key: u64,
    pub depth: u8,
    pub value: i32,
    pub entry_type: TTEntryType,
    pub best_move: Option<Action>,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            depth: 0,
            value: 0,
            entry_type: TTEntryType::Upper,
            best_move: None,
        }
    }
}

pub struct TranspositionTable {
    table: Vec<TTEntry>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        TranspositionTable {
            table: vec![TTEntry::default(); TT_SIZE],
        }
    }

    #[inline(always)]
    pub fn index(&self, key: u64) -> usize {
        (key as usize) & (TT_SIZE - 1)
    }

    pub fn store(&mut self, key: u64, depth: u8, value: i32, entry_type: TTEntryType, best_move: Option<Action>) {
        let idx = self.index(key);
        let entry = &mut self.table[idx];
        if entry.key == 0 || depth >= entry.depth || entry.key != key {
            entry.key = key;
            entry.depth = depth;
            entry.value = value;
            entry.entry_type = entry_type;
            entry.best_move = best_move;
        }
    }

    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let idx = self.index(key);
        let entry = &self.table[idx];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }
}

pub mod book {
    use super::*;
    use std::collections::HashMap;

    pub struct OpeningBook {
        book: HashMap<u64, Vec<(Action, i32)>>,
    }

    impl OpeningBook {
        pub fn new() -> Self {
            let mut book = OpeningBook {
                book: HashMap::new(),
            };
            book.init_all_openings();
            book
        }

        fn init_all_openings(&mut self) {
            self.init_dang_tou_pao();
            self.init_shun_pao();
            self.init_lie_pao();
            self.init_fei_xiang();
            self.init_qi_ma();
            self.init_guo_gong_pao();
            self.init_xian_ren_zhi_lu();
        }

        fn init_dang_tou_pao(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 红方炮二平五 (7,7)->(4,7)
            let a1 = Action::new(Coord::new(7, 7), Coord::new(4, 7), None);
            self.book.insert(key, vec![(a1, 1000)]);

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方马8进7 (7,0)->(6,2)
            let a2 = Action::new(Coord::new(7, 0), Coord::new(6, 2), None);
            self.book.insert(key, vec![(a2, 1000)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a3 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);
            self.book.insert(key, vec![(a3, 1000)]);

            board.make_move(a3);
            // 黑方马2进3 (1,0)->(2,2)
            let a4 = Action::new(Coord::new(1, 0), Coord::new(2, 2), None);
            self.book.insert(board.zobrist_key, vec![(a4, 1000)]);

            board.make_move(a4);
            // 红方车九平八 (0,9)->(0,8)
            let a5_main = Action::new(Coord::new(0, 9), Coord::new(0, 8), None);
            // 红方兵五进一 (4,6)->(4,5)
            let a5_wuqi = Action::new(Coord::new(4, 6), Coord::new(4, 5), None);
            self.book.insert(board.zobrist_key, vec![(a5_main, 800), (a5_wuqi, 200)]);

            let mut board_main = board.clone();
            board_main.make_move(a5_main);
            // 黑方车1平2 (8,0)->(8,1)
            let a6 = Action::new(Coord::new(8, 0), Coord::new(8, 1), None);
            self.book.insert(board_main.zobrist_key, vec![(a6, 1000)]);
        }

        fn init_shun_pao(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 红方炮二平五 (7,7)->(4,7)
            let a1 = Action::new(Coord::new(7, 7), Coord::new(4, 7), None);
            if !self.book.contains_key(&key) {
                self.book.insert(key, vec![(a1, 500)]);
            }

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方炮8平5 (7,0)->(4,0)
            let a2 = Action::new(Coord::new(7, 0), Coord::new(4, 0), None);
            self.book.insert(key, vec![(a2, 600)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a3 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);
            self.book.insert(key, vec![(a3, 1000)]);

            board.make_move(a3);
            let key = board.zobrist_key;
            // 黑方车1进1 (8,0)->(8,1)
            let a4 = Action::new(Coord::new(8, 0), Coord::new(8, 1), None);
            self.book.insert(key, vec![(a4, 1000)]);

            board.make_move(a4);
            let key = board.zobrist_key;
            // 红方车九平八 (0,9)->(0,8)
            let a5 = Action::new(Coord::new(0, 9), Coord::new(0, 8), None);
            self.book.insert(key, vec![(a5, 1000)]);
        }

        fn init_lie_pao(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let _key = board.zobrist_key;
            // 红方炮二平五 (7,7)->(4,7)
            let a1 = Action::new(Coord::new(7, 7), Coord::new(4, 7), None);

            board.make_move(a1);
            let _key = board.zobrist_key;
            // 黑方马8进7 (7,0)->(6,2)
            let a2 = Action::new(Coord::new(7, 0), Coord::new(6, 2), None);

            board.make_move(a2);
            let _key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a3 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);

            board.make_move(a3);
            let key = board.zobrist_key;
            // 黑方炮2平5 (1,0)->(4,0)
            let a4_lie = Action::new(Coord::new(1, 0), Coord::new(4, 0), None);
            // 黑方马2进3 (1,0)->(2,2)
            let a4_normal = Action::new(Coord::new(1, 0), Coord::new(2, 2), None);
            self.book.insert(key, vec![(a4_normal, 700), (a4_lie, 300)]);
        }

        fn init_fei_xiang(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 红方相三进五 (6,9)->(4,7)
            let a1 = Action::new(Coord::new(6, 9), Coord::new(4, 7), None);
            if let Some(mut existing) = self.book.remove(&key) {
                existing.push((a1, 400));
                self.book.insert(key, existing);
            } else {
                self.book.insert(key, vec![(a1, 400)]);
            }

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方炮8平5 (7,0)->(4,0)
            let a2 = Action::new(Coord::new(7, 0), Coord::new(4, 0), None);
            self.book.insert(key, vec![(a2, 600)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a3 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);
            self.book.insert(key, vec![(a3, 1000)]);
        }

        fn init_qi_ma(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a1 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);
            if let Some(existing) = self.book.get_mut(&key) {
                existing.push((a1, 300));
            } else {
                self.book.insert(key, vec![(a1, 300)]);
            }

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方卒7进1 (6,3)->(6,4)
            let a2 = Action::new(Coord::new(6, 3), Coord::new(6, 4), None);
            self.book.insert(key, vec![(a2, 1000)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 修正：红方兵三进一 (6,6)->(6,5)
            let a3 = Action::new(Coord::new(6, 6), Coord::new(6, 5), None);
            self.book.insert(key, vec![(a3, 1000)]);
        }

        fn init_guo_gong_pao(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 红方炮八平七 (1,7)->(3,7)
            let a1 = Action::new(Coord::new(1, 7), Coord::new(3, 7), None);
            if let Some(existing) = self.book.get_mut(&key) {
                existing.push((a1, 200));
            } else {
                self.book.insert(key, vec![(a1, 200)]);
            }

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方马8进7 (7,0)->(6,2)
            let a2 = Action::new(Coord::new(7, 0), Coord::new(6, 2), None);
            self.book.insert(key, vec![(a2, 1000)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 红方马八进七 (1,9)->(2,7)
            let a3 = Action::new(Coord::new(1, 9), Coord::new(2, 7), None);
            self.book.insert(key, vec![(a3, 1000)]);
        }

        fn init_xian_ren_zhi_lu(&mut self) {
            let mut board = Board::new(RuleSet::Official, 1);
            let key = board.zobrist_key;
            // 修正：红方兵三进一 (6,6)->(6,5)
            let a1 = Action::new(Coord::new(6, 6), Coord::new(6, 5), None);
            if let Some(existing) = self.book.get_mut(&key) {
                existing.iter_mut().for_each(|(_, w)| *w = *w * 2 / 3);
                existing.push((a1, 1000));
            } else {
                self.book.insert(key, vec![(a1, 1000)]);
            }

            board.make_move(a1);
            let key = board.zobrist_key;
            // 黑方卒7进1 (6,3)->(6,4)
            let a2 = Action::new(Coord::new(6, 3), Coord::new(6, 4), None);
            self.book.insert(key, vec![(a2, 700)]);

            board.make_move(a2);
            let key = board.zobrist_key;
            // 红方炮八平五 (1,7)->(4,7)
            let a3 = Action::new(Coord::new(1, 7), Coord::new(4, 7), None);
            self.book.insert(key, vec![(a3, 1000)]);
        }

        pub fn probe(&self, board: &Board) -> Option<Action> {
            if let Some(moves) = self.book.get(&board.zobrist_key) {
                if moves.is_empty() {
                    return None;
                }

                let max_weight = moves.iter().map(|(_, w)| *w).max().unwrap_or(0);
                let candidates: Vec<&(Action, i32)> = moves.iter().filter(|(_, w)| *w == max_weight).collect();

                if candidates.is_empty() {
                    return None;
                }

                if candidates.len() == 1 {
                    return Some(candidates[0].0);
                }

                let seed = (board.zobrist_key & 0xFFFF) as usize + (board.move_history.len() % 2) * 0x8000;
                let idx = seed % candidates.len();
                Some(candidates[idx].0)
            } else {
                None
            }
        }
    }

    pub struct EndgameTablebase;

    impl EndgameTablebase {
        #[inline(always)]
        fn check_double_chariot_vs_single(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Chariot as usize] == 2
                && red_other == 2
                && black[PieceType::King as usize] == 1
                && black[PieceType::Chariot as usize] == 1
                && black_other >= 1 && black_other <= 4
            {
                let score = 85000;
                return Some(if side == Color::Red { score } else { -score });
            }
            None
        }

        #[inline(always)]
        fn check_chariot_cannon_vs_chariot(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Chariot as usize] == 1
                && red[PieceType::Cannon as usize] == 1
                && red_other == 2
                && black[PieceType::King as usize] == 1
                && black[PieceType::Chariot as usize] == 1
                && black_other == 1
            {
                let score = 78000;
                return Some(if side == Color::Red { score } else { -score });
            }
            None
        }

        #[inline(always)]
        fn check_pawn_vs_advisor(board: &Board, red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Pawn as usize] == 1
                && red_other == 1
                && black[PieceType::King as usize] == 1
                && black[PieceType::Advisor as usize] == 1
                && black_other == 1
            {
                let mut pawn_pos = None;
                for y in 0..10 {
                    for x in 0..9 {
                        if let Some(p) = board.cells[y][x] {
                            if p.color == Color::Red && p.piece_type == PieceType::Pawn {
                                pawn_pos = Some(Coord::new(x as i8, y as i8));
                                break;
                            }
                        }
                    }
                }

                if let Some(pos) = pawn_pos {
                    if pos.crosses_river(Color::Red) {
                        let score = 80000;
                        return Some(if side == Color::Red { score } else { -score });
                    }
                }
            }
            None
        }

        #[inline(always)]
        fn check_horse_cannon_vs_double_advisor(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Horse as usize] == 1
                && red[PieceType::Cannon as usize] == 1
                && red_other == 2
                && black[PieceType::King as usize] == 1
                && black[PieceType::Advisor as usize] == 2
                && black_other == 2
            {
                let score = 72000;
                return Some(if side == Color::Red { score } else { -score });
            }
            None
        }

        #[inline(always)]
        fn check_horse_vs_advisor(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Horse as usize] == 1
                && red_other == 1
                && black[PieceType::King as usize] == 1
                && black[PieceType::Advisor as usize] == 1
                && black_other == 1
            {
                let score = 68000;
                return Some(if side == Color::Red { score } else { -score });
            }
            None
        }

        #[inline(always)]
        fn check_cannon_advisor_vs_advisor(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Cannon as usize] == 1
                && red[PieceType::Advisor as usize] == 1
                && red_other == 2
                && black[PieceType::King as usize] == 1
                && black[PieceType::Advisor as usize] == 1
                && black_other == 1
            {
                let score = 70000;
                return Some(if side == Color::Red { score } else { -score });
            }
            None
        }

        #[inline(always)]
        fn check_chariot_vs_defense(red: &[i32; 7], black: &[i32; 7], red_other: i32, black_other: i32, side: Color) -> Option<i32> {
            if red[PieceType::King as usize] == 1
                && red[PieceType::Chariot as usize] == 1
                && red_other == 1
                && black[PieceType::King as usize] == 1
            {
                if (black[PieceType::Horse as usize] == 1 && black_other == 1)
                    || (black[PieceType::Cannon as usize] == 1 && black_other == 1)
                    || (black[PieceType::Advisor as usize] == 2 && black_other == 2)
                    || (black[PieceType::Elephant as usize] == 2 && black_other == 2)
                {
                    let score = 75000;
                    return Some(if side == Color::Red { score } else { -score });
                }
            }
            None
        }

        pub fn probe(board: &Board, side: Color) -> Option<i32> {
            let mut red = [0; 7];
            let mut black = [0; 7];
            for y in 0..10 {
                for x in 0..9 {
                    if let Some(p) = board.cells[y][x] {
                        match p.color {
                            Color::Red => red[p.piece_type as usize] += 1,
                            Color::Black => black[p.piece_type as usize] += 1,
                        }
                    }
                }
            }

            let red_other = red.iter().skip(1).sum::<i32>();
            let black_other = black.iter().skip(1).sum::<i32>();

            if let Some(score) = Self::check_double_chariot_vs_single(&red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_chariot_cannon_vs_chariot(&red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_pawn_vs_advisor(board, &red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_horse_cannon_vs_double_advisor(&red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_horse_vs_advisor(&red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_cannon_advisor_vs_advisor(&red, &black, red_other, black_other, side) {
                return Some(score);
            }
            if let Some(score) = Self::check_chariot_vs_defense(&red, &black, red_other, black_other, side) {
                return Some(score);
            }

            None
        }
    }
}

pub mod movegen {
    use super::*;

    #[inline(always)]
    fn is_valid_target(board: &Board, tar: Coord, color: Color) -> bool {
        match board.get(tar) {
            Some(p) => p.color != color,
            None => true,
        }
    }

    pub fn generate_pawn_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dir = if color == Color::Red { -1 } else { 1 };

        let forward = Coord::new(pos.x, pos.y + dir);
        if forward.is_valid() && is_valid_target(board, forward, color) {
            moves.push(forward);
        }

        if pos.crosses_river(color) {
            for dx in [-1, 1] {
                let side = Coord::new(pos.x + dx, pos.y);
                if side.is_valid() && is_valid_target(board, side, color) {
                    moves.push(side);
                }
            }
        }

        moves
    }

    pub fn generate_horse_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(2,1,1,0), (2,-1,1,0), (-2,1,-1,0), (-2,-1,-1,0),
            (1,2,0,1), (1,-2,0,-1), (-1,2,0,1), (-1,-2,0,-1)];

        for (dx, dy, bx, by) in deltas {
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            let block = Coord::new(pos.x + bx, pos.y + by);

            if tar.is_valid() && board.get(block).is_none() && is_valid_target(board, tar, color) {
                moves.push(tar);
            }
        }

        moves
    }

    pub fn generate_chariot_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dirs = [(0,1), (0,-1), (1,0), (-1,0)];

        for (dx, dy) in dirs {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x, y);
                match board.get(tar) {
                    Some(p) => {
                        if p.color != color {
                            moves.push(tar);
                        }
                        break;
                    }
                    None => moves.push(tar),
                }
                x += dx;
                y += dy;
            }
        }

        moves
    }

    pub fn generate_cannon_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dirs = [(0,1), (0,-1), (1,0), (-1,0)];

        for (dx, dy) in dirs {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;
            let mut jumped = false;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x, y);
                match board.get(tar) {
                    Some(p) => {
                        if jumped {
                            if p.color != color {
                                moves.push(tar);
                            }
                            break;
                        } else {
                            jumped = true;
                        }
                    }
                    None => {
                        if !jumped {
                            moves.push(tar);
                        }
                    }
                }
                x += dx;
                y += dy;
            }
        }

        moves
    }

    pub fn generate_elephant_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(2,2,1,1), (2,-2,1,-1), (-2,2,-1,1), (-2,-2,-1,-1)];

        for (dx, dy, bx, by) in deltas {
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            let block = Coord::new(pos.x + bx, pos.y + by);

            if tar.is_valid() && !tar.crosses_river(color) && board.get(block).is_none() && is_valid_target(board, tar, color) {
                moves.push(tar);
            }
        }

        moves
    }

    pub fn generate_advisor_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(1,1), (1,-1), (-1,1), (-1,-1)];

        for (dx, dy) in deltas {
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            if tar.is_valid() && tar.in_palace(color) && is_valid_target(board, tar, color) {
                moves.push(tar);
            }
        }

        moves
    }

    pub fn generate_king_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(0,1), (0,-1), (1,0), (-1,0)];

        for (dx, dy) in deltas {
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            if tar.is_valid() && tar.in_palace(color) && is_valid_target(board, tar, color) {
                moves.push(tar);
            }
        }

        moves
    }

    pub fn generate_pseudo_moves(board: &Board, color: Color) -> Vec<Action> {
        let mut moves = Vec::with_capacity(60);

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = board.get(pos) {
                    if piece.color == color {
                        let targets = match piece.piece_type {
                            PieceType::Pawn => generate_pawn_moves(board, pos, color),
                            PieceType::Horse => generate_horse_moves(board, pos, color),
                            PieceType::Chariot => generate_chariot_moves(board, pos, color),
                            PieceType::Cannon => generate_cannon_moves(board, pos, color),
                            PieceType::Elephant => generate_elephant_moves(board, pos, color),
                            PieceType::Advisor => generate_advisor_moves(board, pos, color),
                            PieceType::King => generate_king_moves(board, pos, color),
                        };

                        for tar in targets {
                            moves.push(Action::new(pos, tar, board.get(tar)));
                        }
                    }
                }
            }
        }

        moves
    }

    #[inline(always)]
    fn is_legal_move(board: &mut Board, action: Action, side: Color) -> (bool, bool) {
        let src = action.src;
        let tar = action.tar;
        let piece = board.get(src).unwrap();
        let captured = action.captured;

        board.cells[tar.y as usize][tar.x as usize] = Some(piece);
        board.cells[src.y as usize][src.x as usize] = None;

        let is_self_checked = board.is_check(side);
        let legal = !is_self_checked;
        let gives_check = legal && board.is_check(side.opponent());

        board.cells[src.y as usize][src.x as usize] = Some(piece);
        board.cells[tar.y as usize][tar.x as usize] = captured;

        (legal, gives_check)
    }

    pub fn generate_legal_moves(board: &mut Board, color: Color) -> Vec<Action> {
        let mut legal_moves = Vec::with_capacity(40);
        let pseudo_moves = generate_pseudo_moves(board, color);

        for mut action in pseudo_moves {
            let (legal, gives_check) = is_legal_move(board, action, color);
            if legal {
                action.is_check = gives_check;
                legal_moves.push(action);
            }
        }

        legal_moves
    }

    pub fn generate_capture_moves(board: &mut Board, color: Color) -> Vec<Action> {
        let mut moves = generate_pseudo_moves(board, color);
        moves.retain(|a| a.captured.is_some());

        let mut legal_captures = Vec::new();
        for mut action in moves {
            let (legal, gives_check) = is_legal_move(board, action, color);
            if legal {
                action.is_check = gives_check;
                legal_captures.push(action);
            }
        }

        legal_captures
    }

    const SEE_VALUE: [i32; 7] = [10000, 110, 110, 70, 320, 320, 600];

    pub fn see(board: &Board, src: Coord, tar: Coord) -> i32 {
        let mut swap_list = [0; 32];
        let mut swap_idx = 0;
        let mut side = board.get(src).unwrap().color;
        let current_attacker = src;
        let captured_value = SEE_VALUE[board.get(tar).map_or(0, |p| p.piece_type as usize)];

        swap_list[swap_idx] = captured_value;
        swap_idx += 1;

        let mut board_copy = board.clone();
        let moving_piece = board_copy.get(current_attacker).unwrap();
        board_copy.set_internal(tar, Some(moving_piece));
        board_copy.set_internal(current_attacker, None);
        side = side.opponent();

        loop {
            let (attacker, attacker_value) = find_least_valuable_attacker(&board_copy, tar, side);
            if attacker.is_none() {
                break;
            }

            swap_list[swap_idx] = -swap_list[swap_idx - 1] + attacker_value;
            swap_idx += 1;

            let attacker_piece = board_copy.get(attacker.unwrap()).unwrap();
            board_copy.set_internal(tar, Some(attacker_piece));
            board_copy.set_internal(attacker.unwrap(), None);
            side = side.opponent();

            if attacker_piece.piece_type == PieceType::King {
                break;
            }
        }

        let mut score = 0;
        for i in (0..swap_idx).rev() {
            score = (-score).max(swap_list[i]);
        }

        score
    }

    fn find_least_valuable_attacker(board: &Board, tar: Coord, side: Color) -> (Option<Coord>, i32) {
        let mut min_value = i32::MAX;
        let mut min_attacker = None;

        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = board.get(pos) {
                    if piece.color != side {
                        continue;
                    }
                    let can_attack = match piece.piece_type {
                        PieceType::Pawn => generate_pawn_moves(board, pos, side).contains(&tar),
                        PieceType::Horse => generate_horse_moves(board, pos, side).contains(&tar),
                        PieceType::Chariot => generate_chariot_moves(board, pos, side).contains(&tar),
                        PieceType::Cannon => generate_cannon_moves(board, pos, side).contains(&tar),
                        PieceType::Elephant => generate_elephant_moves(board, pos, side).contains(&tar),
                        PieceType::Advisor => generate_advisor_moves(board, pos, side).contains(&tar),
                        PieceType::King => generate_king_moves(board, pos, side).contains(&tar),
                    };

                    if can_attack {
                        let value = SEE_VALUE[piece.piece_type as usize];
                        if value < min_value {
                            min_value = value;
                            min_attacker = Some(pos);
                        }
                    }
                }
            }
        }

        (min_attacker, min_value)
    }
}

#[derive(Clone)]
pub struct Board {
    pub cells: [[Option<Piece>; 9]; 10],
    pub zobrist_key: u64,
    pub current_side: Color,
    pub rule_set: RuleSet,
    pub move_history: Vec<Action>,
    pub repetition_history: HashMap<u64, u8>,
}

impl Board {
    pub fn new(rule_set: RuleSet, order: u8) -> Self {
        let mut cells = [[None; 9]; 10];
        let back_row = [
            PieceType::Chariot,
            PieceType::Horse,
            PieceType::Elephant,
            PieceType::Advisor,
            PieceType::King,
            PieceType::Advisor,
            PieceType::Elephant,
            PieceType::Horse,
            PieceType::Chariot,
        ];

        let place_row = |cells: &mut [[Option<Piece>; 9]; 10], y: usize, color: Color| {
            for (x, &pt) in back_row.iter().enumerate() {
                cells[y][x] = Some(Piece { color, piece_type: pt });
            }
        };
        let place_cannon = |cells: &mut [[Option<Piece>; 9]; 10], y: usize, color: Color| {
            cells[y][1] = Some(Piece { color, piece_type: PieceType::Cannon });
            cells[y][7] = Some(Piece { color, piece_type: PieceType::Cannon });
        };
        let place_pawns = |cells: &mut [[Option<Piece>; 9]; 10], y: usize, color: Color| {
            for &x in &[0, 2, 4, 6, 8] {
                cells[y][x] = Some(Piece { color, piece_type: PieceType::Pawn });
            }
        };

        place_row(&mut cells, 0, Color::Black);
        place_cannon(&mut cells, 2, Color::Black);
        place_pawns(&mut cells, 3, Color::Black);
        place_row(&mut cells, 9, Color::Red);
        place_cannon(&mut cells, 7, Color::Red);
        place_pawns(&mut cells, 6, Color::Red);

        let zobrist = get_zobrist();
        let mut zobrist_key = 0;
        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = cells[y][x] {
                    let pos_idx = zobrist.pos_idx(Coord::new(x as i8, y as i8));
                    zobrist_key ^= zobrist.pieces[pos_idx][p.color as usize][p.piece_type as usize];
                }
            }
        }
        zobrist_key ^= zobrist.side;

        let mut repetition_history = HashMap::new();
        repetition_history.insert(zobrist_key, 1);

        Board {
            cells,
            zobrist_key,
            current_side: match order { 1 => Color::Red, 2 => Color::Black, _=>unreachable!() },
            rule_set,
            move_history: Vec::with_capacity(200),
            repetition_history,
        }
    }

    #[inline(always)]
    pub fn get(&self, coord: Coord) -> Option<Piece> {
        if coord.is_valid() {
            self.cells[coord.y as usize][coord.x as usize]
        } else {
            None
        }
    }

    #[inline(always)]
    fn set_internal(&mut self, coord: Coord, piece: Option<Piece>) {
        if !coord.is_valid() {
            return;
        }
        let zobrist = get_zobrist();
        let pos_idx = zobrist.pos_idx(coord);

        if let Some(old_p) = self.cells[coord.y as usize][coord.x as usize] {
            self.zobrist_key ^= zobrist.pieces[pos_idx][old_p.color as usize][old_p.piece_type as usize];
        }
        if let Some(new_p) = piece {
            self.zobrist_key ^= zobrist.pieces[pos_idx][new_p.color as usize][new_p.piece_type as usize];
        }

        self.cells[coord.y as usize][coord.x as usize] = piece;
    }

    pub fn make_move(&mut self, mut action: Action) {
        // 安全检查：保留警告输出
        let Some(piece) = self.get(action.src) else {
            eprintln!("Warning: Invalid move from empty square {:?}", action.src);
            return;
        };

        self.set_internal(action.tar, Some(piece));
        self.set_internal(action.src, None);

        let zobrist = get_zobrist();
        self.zobrist_key ^= zobrist.side;
        let prev_side = self.current_side;
        self.current_side = self.current_side.opponent();

        action.is_check = self.is_check(self.current_side);
        action.is_capture_threat = self.is_capture_threat_internal(prev_side);

        self.move_history.push(action);
        *self.repetition_history.entry(self.zobrist_key).or_insert(0) += 1;
    }

    pub fn undo_move(&mut self, action: Action) {
        *self.repetition_history.get_mut(&self.zobrist_key).unwrap() -= 1;
        if self.repetition_history[&self.zobrist_key] == 0 {
            self.repetition_history.remove(&self.zobrist_key);
        }

        let piece = self.get(action.tar).unwrap();
        self.set_internal(action.src, Some(piece));
        self.set_internal(action.tar, action.captured);

        let zobrist = get_zobrist();
        self.zobrist_key ^= zobrist.side;
        self.current_side = self.current_side.opponent();

        self.move_history.pop();
    }

    #[inline(always)]
    pub fn find_kings(&self) -> (Option<Coord>, Option<Coord>) {
        let mut red_king = None;
        let mut black_king = None;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = self.cells[y][x] {
                    if p.piece_type == PieceType::King {
                        let coord = Coord::new(x as i8, y as i8);
                        match p.color {
                            Color::Red => red_king = Some(coord),
                            Color::Black => black_king = Some(coord),
                        }
                    }
                }
            }
        }

        (red_king, black_king)
    }

    #[inline(always)]
    pub fn is_face_to_face(&self) -> bool {
        let (red_king, black_king) = self.find_kings();
        let (rk, bk) = match (red_king, black_king) {
            (Some(r), Some(b)) => (r, b),
            _ => return false,
        };

        if rk.x != bk.x {
            return false;
        }

        let min_y = rk.y.min(bk.y);
        let max_y = rk.y.max(bk.y);
        for y in (min_y + 1)..max_y {
            if self.cells[y as usize][rk.x as usize].is_some() {
                return false;
            }
        }

        true
    }

    pub fn is_check(&self, color: Color) -> bool {
        let (rk, bk) = self.find_kings();
        let king_pos = match color {
            Color::Red => match rk {
                Some(pos) => pos,
                None => return false,
            },
            Color::Black => match bk {
                Some(pos) => pos,
                None => return false,
            },
        };
        let opponent = color.opponent();

        if self.is_face_to_face() {
            return true;
        }

        let pawn_dir = if color == Color::Red { -1 } else { 1 };
        let forward = Coord::new(king_pos.x, king_pos.y + pawn_dir);
        if let Some(p) = self.get(forward) {
            if p.color == opponent && p.piece_type == PieceType::Pawn {
                return true;
            }
        }
        for dx in [-1, 1] {
            let side = Coord::new(king_pos.x + dx, king_pos.y);
            if let Some(p) = self.get(side) {
                if p.color == opponent && p.piece_type == PieceType::Pawn {
                    return true;
                }
            }
        }

        let horse_deltas = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)];
        let block_deltas = [(1,0), (1,0), (-1,0), (-1,0), (0,1), (0,-1), (0,1), (0,-1)];
        for i in 0..8 {
            let (dx, dy) = horse_deltas[i];
            let (bx, by) = block_deltas[i];
            let pos = Coord::new(king_pos.x + dx, king_pos.y + dy);
            let block = Coord::new(king_pos.x + bx, king_pos.y + by);

            if pos.is_valid() && self.get(block).is_none() {
                if let Some(p) = self.get(pos) {
                    if p.color == opponent && p.piece_type == PieceType::Horse {
                        return true;
                    }
                }
            }
        }

        let dirs = [(0,1), (0,-1), (1,0), (-1,0)];
        for (dx, dy) in dirs {
            let mut x = king_pos.x + dx;
            let mut y = king_pos.y + dy;
            let mut jumped = false;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let pos = Coord::new(x, y);
                if let Some(p) = self.get(pos) {
                    if p.color == opponent {
                        if !jumped && p.piece_type == PieceType::Chariot {
                            return true;
                        }
                        if jumped && p.piece_type == PieceType::Cannon {
                            return true;
                        }
                    }
                    break;
                }
                jumped = true;
                x += dx;
                y += dy;
            }
        }

        false
    }

    fn is_capture_threat_internal(&self, attacker_color: Color) -> bool {
        let opponent = attacker_color.opponent();

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = self.get(pos) {
                    if piece.color == attacker_color {
                        let targets = match piece.piece_type {
                            PieceType::Pawn => movegen::generate_pawn_moves(self, pos, attacker_color),
                            PieceType::Horse => movegen::generate_horse_moves(self, pos, attacker_color),
                            PieceType::Chariot => movegen::generate_chariot_moves(self, pos, attacker_color),
                            PieceType::Cannon => movegen::generate_cannon_moves(self, pos, attacker_color),
                            PieceType::Elephant => movegen::generate_elephant_moves(self, pos, attacker_color),
                            PieceType::Advisor => movegen::generate_advisor_moves(self, pos, attacker_color),
                            PieceType::King => movegen::generate_king_moves(self, pos, attacker_color),
                        };

                        for tar in targets {
                            if let Some(target_piece) = self.get(tar) {
                                if target_piece.color == opponent && target_piece.piece_type != PieceType::King {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }

    pub fn is_repetition_violation(&self) -> Option<Color> {
        if matches!(self.rule_set, RuleSet::NoRestriction) {
            return None;
        }

        for (&_key, &count) in &self.repetition_history {
            if count >= REPETITION_VIOLATION_COUNT {
                let cycle_moves: Vec<&Action> = self.move_history
                    .iter()
                    .rev()
                    .take((REPETITION_VIOLATION_COUNT * 2) as usize)
                    .collect();

                if self.rule_set.is_long_check_banned() {
                    let red_long_check = cycle_moves.iter().step_by(2).any(|a| a.is_check);
                    let black_long_check = cycle_moves.iter().skip(1).step_by(2).any(|a| a.is_check);

                    if red_long_check && !black_long_check {
                        return Some(Color::Black);
                    }
                    if black_long_check && !red_long_check {
                        return Some(Color::Red);
                    }
                }

                if self.rule_set.is_long_capture_banned() {
                    let red_long_capture = cycle_moves.iter().step_by(2).any(|a| a.is_capture_threat);
                    let black_long_capture = cycle_moves.iter().skip(1).step_by(2).any(|a| a.is_capture_threat);

                    if red_long_capture && !black_long_capture {
                        return Some(Color::Black);
                    }
                    if black_long_capture && !red_long_capture {
                        return Some(Color::Red);
                    }
                }

                return None;
            }
        }

        None
    }

    pub fn get_winner(&self) -> Option<Color> {
        let (red_king, black_king) = self.find_kings();
        match (red_king, black_king) {
            (Some(_), Some(_)) => None,
            (Some(_), None) => Some(Color::Red),
            (None, Some(_)) => Some(Color::Black),
            (None, None) => None,
        }
    }

    pub fn flip_vertically(&mut self) {
        for x in 0..9 {
            for y in 0..5 {
                let tmp = self.cells[y][x];
                self.cells[y][x] = self.cells[9 - y][x];
                self.cells[9 - y][x] = tmp;
            }
        }
    }
}

const PIECE_CHARS: [[char; 7]; 2] = [
    ['帥', '仕', '相', '兵', '馬', '炮', '車'],
    ['將', '士', '象', '卒', '馬', '砲', '車'],
];

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "  0 1 2 3 4 5 6 7 8")?;
        writeln!(f, "  -------------------")?;
        for y in 0..10 {
            write!(f, "{}|", y)?;
            for x in 0..9 {
                let c = match self.cells[y][x] {
                    Some(p) => PIECE_CHARS[p.color as usize][p.piece_type as usize],
                    None => '·',
                };
                write!(f, "{} ", c)?;
            }
            writeln!(f, "|{}", y)?;
        }
        writeln!(f, "  -------------------")?;
        writeln!(f, "  0 1 2 3 4 5 6 7 8")
    }
}

pub mod eval {
    use super::*;
    use super::book::EndgameTablebase;

    const MG_VALUE: [i32; 7] = [10000, 120, 120, 80, 350, 350, 650];
    const EG_VALUE: [i32; 7] = [10000, 120, 120, 200, 300, 250, 700];

    const MG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 30, 50, 30, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 30, 50, 30, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
    ];

    const MG_PST_ADVISOR: [[i32; 9]; 10] = [
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 40, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
        [0, 0, 0, 0, 40, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 30, 0, 0, 0],
    ];

    const MG_PST_ELEPHANT: [[i32; 9]; 10] = [
        [0, 0, 10, 0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 20, 0, 30, 0, 20, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 25, 0, 40, 0, 25, 0, 0],
        [0, 0, 25, 0, 40, 0, 25, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 20, 0, 30, 0, 20, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 10, 0, 0, 0, 10, 0, 0],
    ];

    const MG_PST_HORSE: [[i32; 9]; 10] = [
        [5, 10, 20, 20, 20, 20, 20, 10, 5],
        [5, 15, 30, 40, 50, 40, 30, 15, 5],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [20, 40, 60, 80, 90, 80, 60, 40, 20],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [20, 40, 60, 80, 90, 80, 60, 40, 20],
        [10, 30, 50, 70, 80, 70, 50, 30, 10],
        [5, 15, 30, 40, 50, 40, 30, 15, 5],
        [5, 10, 20, 20, 20, 20, 20, 10, 5],
    ];

    const MG_PST_CHARIOT: [[i32; 9]; 10] = [
        [10, 20, 30, 40, 50, 40, 30, 20, 10],
        [20, 30, 40, 50, 60, 50, 40, 30, 20],
        [30, 40, 50, 60, 70, 60, 50, 40, 30],
        [40, 50, 60, 80, 90, 80, 60, 50, 40],
        [30, 40, 50, 70, 80, 70, 50, 40, 30],
        [30, 40, 50, 70, 80, 70, 50, 40, 30],
        [40, 50, 60, 80, 90, 80, 60, 50, 40],
        [30, 40, 50, 60, 70, 60, 50, 40, 30],
        [20, 30, 40, 50, 60, 50, 40, 30, 20],
        [10, 20, 30, 40, 50, 40, 30, 20, 10],
    ];

    const MG_PST_CANNON: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 30, 40, 50, 40, 30, 20, 10],
        [30, 40, 50, 60, 70, 60, 50, 40, 30],
        [40, 50, 60, 80, 90, 80, 60, 50, 40],
        [30, 40, 50, 70, 80, 70, 50, 40, 30],
        [30, 40, 50, 70, 80, 70, 50, 40, 30],
        [40, 50, 60, 80, 90, 80, 60, 50, 40],
        [30, 40, 50, 60, 70, 60, 50, 40, 30],
        [10, 20, 30, 40, 50, 40, 30, 20, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const MG_PST_PAWN: [[i32; 9]; 10] = [
        [140, 160, 180, 200, 220, 200, 180, 160, 140],
        [120, 140, 160, 180, 200, 180, 160, 140, 120],
        [100, 120, 140, 160, 180, 160, 140, 120, 100],
        [80, 100, 120, 140, 160, 140, 120, 100, 80],
        [60, 80, 100, 120, 140, 120, 100, 80, 60],
        [30, 40, 50, 60, 70, 60, 50, 40, 30],
        [10, 20, 30, 40, 50, 40, 30, 20, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const EG_PST_KING: [[i32; 9]; 10] = [
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 30, 50, 30, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
        [0, 0, 0, 30, 50, 30, 0, 0, 0],
        [0, 0, 0, 20, 40, 20, 0, 0, 0],
    ];

    const EG_PST_PAWN: [[i32; 9]; 10] = [
        [300, 350, 400, 450, 500, 450, 400, 350, 300],
        [280, 320, 380, 420, 480, 420, 380, 320, 280],
        [250, 300, 350, 400, 450, 400, 350, 300, 250],
        [200, 250, 300, 350, 400, 350, 300, 250, 200],
        [150, 200, 250, 300, 350, 300, 250, 200, 150],
        [100, 120, 150, 180, 200, 180, 150, 120, 100],
        [50, 80, 100, 120, 150, 120, 100, 80, 50],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const PHASE_WEIGHTS: [i32; 7] = [0, 1, 1, 1, 4, 4, 8];
    const TOTAL_PHASE: i32 = 2 * (1 + 1 + 1 + 4 + 4 + 8);

    fn game_phase(board: &Board) -> f32 {
        let mut phase = 0;
        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = board.cells[y][x] {
                    phase += PHASE_WEIGHTS[p.piece_type as usize];
                }
            }
        }
        (phase as f32 / TOTAL_PHASE as f32).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn pst_val(
        pt: PieceType,
        color: Color,
        x: usize,
        y: usize,
        phase: f32,
    ) -> i32 {
        let y_mirrored = 9 - y;
        let (y_mg, y_eg) = if color == Color::Red {
            (y, y)
        } else {
            (y_mirrored, y_mirrored)
        };

        let mg = match pt {
            PieceType::King => MG_PST_KING[y_mg][x],
            PieceType::Advisor => MG_PST_ADVISOR[y_mg][x],
            PieceType::Elephant => MG_PST_ELEPHANT[y_mg][x],
            PieceType::Horse => MG_PST_HORSE[y_mg][x],
            PieceType::Chariot => MG_PST_CHARIOT[y_mg][x],
            PieceType::Cannon => MG_PST_CANNON[y_mg][x],
            PieceType::Pawn => MG_PST_PAWN[y_mg][x],
        };

        let eg = match pt {
            PieceType::King => EG_PST_KING[y_eg][x],
            PieceType::Pawn => EG_PST_PAWN[y_eg][x],
            _ => mg,
        };

        (mg as f32 * phase + eg as f32 * (1.0 - phase)) as i32
    }

    fn center_control(board: &Board, color: Color, phase: f32) -> i32 {
        let mut control = 0;
        let _opponent = color.opponent();

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if !pos.in_core_area(color) {
                    continue;
                }

                let mut our_control = 0;
                let mut their_control = 0;

                for y2 in 0..10 {
                    for x2 in 0..9 {
                        let attacker_pos = Coord::new(x2 as i8, y2 as i8);
                        if let Some(piece) = board.get(attacker_pos) {
                            let can_attack = match piece.piece_type {
                                PieceType::Pawn => movegen::generate_pawn_moves(board, attacker_pos, piece.color).contains(&pos),
                                PieceType::Horse => movegen::generate_horse_moves(board, attacker_pos, piece.color).contains(&pos),
                                PieceType::Chariot => movegen::generate_chariot_moves(board, attacker_pos, piece.color).contains(&pos),
                                PieceType::Cannon => movegen::generate_cannon_moves(board, attacker_pos, piece.color).contains(&pos),
                                _ => false,
                            };

                            if can_attack {
                                let weight = match piece.piece_type {
                                    PieceType::Chariot | PieceType::Cannon => 3,
                                    PieceType::Horse | PieceType::Pawn => 2,
                                    _ => 0,
                                };
                                if piece.color == color {
                                    our_control += weight;
                                } else {
                                    their_control += weight;
                                }
                            }
                        }
                    }
                }

                control += (our_control - their_control) * color.sign();
            }
        }

        (control as f32 * (0.5 + 0.5 * phase)) as i32
    }

    fn piece_coordination(board: &Board, color: Color, phase: f32) -> i32 {
        let mut coordination = 0;
        let (rk, bk) = board.find_kings();
        let _our_king = match color {
            Color::Red => rk,
            Color::Black => bk,
        };
        let enemy_king = match color {
            Color::Red => bk,
            Color::Black => rk,
        };

        let mut our_pieces = Vec::new();
        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(piece) = board.get(pos) {
                    if piece.color == color {
                        our_pieces.push((piece, pos));
                    }
                }
            }
        }

        let mut has_horse = false;
        let mut has_chariot = false;
        let mut horse_in_attack = false;
        let mut chariot_in_support = false;

        for (piece, pos) in &our_pieces {
            match piece.piece_type {
                PieceType::Horse => {
                    has_horse = true;
                    if let Some(ek) = enemy_king {
                        let dist = pos.distance_to(ek);
                        if dist <= 3 && pos.crosses_river(color) {
                            horse_in_attack = true;
                        }
                    }
                }
                PieceType::Chariot => {
                    has_chariot = true;
                    if (pos.x == 3 || pos.x == 4 || pos.x == 5) && pos.in_core_area(color) {
                        chariot_in_support = true;
                    }
                }
                _ => {}
            }
        }

        if has_horse && has_chariot && horse_in_attack && chariot_in_support {
            coordination += 50 * color.sign();
        }

        for (piece, pos) in &our_pieces {
            if piece.piece_type == PieceType::Cannon {
                let dirs = [(0,1), (0,-1), (1,0), (-1,0)];
                for (dx, dy) in dirs {
                    let mut x = pos.x + dx;
                    let mut y = pos.y + dy;
                    let mut platform_found = false;

                    while x >= 0 && x < 9 && y >= 0 && y < 10 {
                        let tar = Coord::new(x, y);
                        if let Some(p) = board.get(tar) {
                            if !platform_found {
                                if p.color == color && (p.piece_type == PieceType::Advisor || p.piece_type == PieceType::Elephant || p.piece_type == PieceType::Pawn) {
                                    coordination += 20 * color.sign();
                                }
                                platform_found = true;
                            } else {
                                break;
                            }
                        }
                        x += dx;
                        y += dy;
                    }
                }
            }
        }

        if our_pieces.len() >= 4 {
            let mut total_dist = 0;
            let mut count = 0;
            for i in 0..our_pieces.len() {
                for j in i+1..our_pieces.len() {
                    total_dist += our_pieces[i].1.distance_to(our_pieces[j].1);
                    count += 1;
                }
            }
            let avg_dist = if count > 0 { total_dist / count } else { 0 };
            let optimal_dist = 6 + (our_pieces.len() as i32 / 2);
            let dist_penalty = (avg_dist - optimal_dist).abs();
            coordination -= dist_penalty * 2 * color.sign();
        }

        (coordination as f32 * (0.3 + 0.7 * phase)) as i32
    }

    fn horse_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
        let deltas = [(2,1,1,0), (2,-1,1,0), (-2,1,-1,0), (-2,-1,-1,0),
            (1,2,0,1), (1,-2,0,-1), (-1,2,0,1), (-1,-2,0,-1)];
        let mut mobility = 0;

        for (dx, dy, bx, by) in deltas {
            let tar = Coord::new(pos.x + dx, pos.y + dy);
            let block = Coord::new(pos.x + bx, pos.y + by);

            if tar.is_valid() && board.get(block).is_none() {
                mobility += 1;
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        mobility += 5;
                    }
                }
            }
        }
        mobility
    }

    fn cannon_support(board: &Board, pos: Coord, color: Color) -> i32 {
        let dirs = [(0,1), (0,-1), (1,0), (-1,0)];
        let mut support = 0;

        for (dx, dy) in dirs {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;
            let mut platform_found = false;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x, y);
                if board.get(tar).is_some() {
                    if !platform_found {
                        platform_found = true;
                    } else {
                        if board.get(tar).unwrap().color != color {
                            support += 10;
                        }
                        break;
                    }
                }
                x += dx;
                y += dy;
            }
        }
        support
    }

    fn king_safety(board: &Board, color: Color, phase: f32) -> Option<i32> {
        let (rk, bk) = board.find_kings();
        let king_pos = match color {
            Color::Red => rk?,
            Color::Black => bk?,
        };
        let opponent = color.opponent();
        let mut safety = 0;

        let palace_deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
        for (dx, dy) in palace_deltas {
            let pos = Coord::new(king_pos.x + dx, king_pos.y + dy);
            if pos.is_valid() && pos.in_palace(color) {
                if let Some(p) = board.get(pos) {
                    if p.color == color {
                        safety += match p.piece_type {
                            PieceType::Advisor => 25,
                            PieceType::Elephant => 15,
                            _ => 5,
                        };
                    }
                }
            }
        }

        let mg_factor = phase;
        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(p) = board.get(pos) {
                    if p.color == opponent {
                        let dist = (pos.x - king_pos.x).abs() + (pos.y - king_pos.y).abs();
                        let threat = match p.piece_type {
                            PieceType::Chariot => (14 - dist).max(0) as i32 * 10,
                            PieceType::Cannon => (12 - dist).max(0) as i32 * 7,
                            PieceType::Horse => (10 - dist).max(0) as i32 * 8,
                            PieceType::Pawn if dist <= 2 => 20,
                            _ => 0,
                        };
                        safety -= (threat as f32 * mg_factor) as i32;
                    }
                }
            }
        }

        Some(safety * color.sign())
    }

    fn chariot_mobility(board: &Board, pos: Coord, color: Color) -> i32 {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut score = 0;

        for (dx, dy) in dirs {
            let mut x = pos.x + dx;
            let mut y = pos.y + dy;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x, y);
                if board.get(tar).is_some() {
                    if board.get(tar).unwrap().color != color {
                        score += 5;
                    }
                    break;
                }
                score += 10;
                x += dx;
                y += dy;
            }
        }
        score
    }

    fn pawn_structure(board: &Board, color: Color, phase: f32) -> i32 {
        let mut score = 0;
        let eg_factor = 1.0 - phase;

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(p) = board.get(pos) {
                    if p.color == color && p.piece_type == PieceType::Pawn {
                        let left = Coord::new(pos.x - 1, pos.y);
                        let right = Coord::new(pos.x + 1, pos.y);
                        let mut linked = 0;
                        if let Some(lp) = board.get(left) {
                            if lp.color == color && lp.piece_type == PieceType::Pawn {
                                linked += 1;
                            }
                        }
                        if let Some(rp) = board.get(right) {
                            if rp.color == color && rp.piece_type == PieceType::Pawn {
                                linked += 1;
                            }
                        }
                        score += linked * (20.0 + 30.0 * eg_factor) as i32;

                        if eg_factor > 0.5 && pos.crosses_river(color) {
                            let (rk, bk) = board.find_kings();
                            let enemy_king = if color == Color::Red { bk } else { rk };
                            if let Some(ek) = enemy_king {
                                let dist = (pos.x - ek.x).abs() + (pos.y - ek.y).abs();
                                if dist <= 2 {
                                    score += (3 - dist) as i32 * 50;
                                }
                            }
                        }
                    }
                }
            }
        }
        score * color.sign()
    }

    fn elephant_structure(board: &Board, color: Color, phase: f32) -> i32 {
        let mut score = 0;
        let mut elephants = Vec::new();

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as i8, y as i8);
                if let Some(p) = board.get(pos) {
                    if p.color == color && p.piece_type == PieceType::Elephant {
                        elephants.push(pos);
                    }
                }
            }
        }

        let count = elephants.len();
        let mg_factor = phase;
        let eg_factor = 1.0 - phase;

        match count {
            0 => {
                score -= (150.0 * mg_factor + 100.0 * eg_factor) as i32;
            }
            1 => {
                score -= (80.0 * mg_factor + 50.0 * eg_factor) as i32;
                if let Some(pos) = elephants.first() {
                    let moves = movegen::generate_elephant_moves(board, *pos, color);
                    let mut protected = false;
                    for m in moves {
                        if let Some(p) = board.get(m) {
                            if p.color == color && (p.piece_type == PieceType::Advisor || p.piece_type == PieceType::Pawn) {
                                protected = true;
                                break;
                            }
                        }
                    }
                    if !protected {
                        score -= 30;
                    }
                }
            }
            2 => {
                let pos1 = elephants[0];
                let pos2 = elephants[1];
                let moves1 = movegen::generate_elephant_moves(board, pos1, color);
                let moves2 = movegen::generate_elephant_moves(board, pos2, color);

                let mut linked = false;
                for m1 in &moves1 {
                    if moves2.contains(m1) {
                        linked = true;
                        break;
                    }
                }

                if linked {
                    score += (60.0 * mg_factor + 40.0 * eg_factor) as i32;
                }
            }
            _ => {}
        }

        score * color.sign()
    }

    pub fn evaluate(board: &Board, side: Color) -> i32 {
        if let Some(score) = EndgameTablebase::probe(board, side) {
            return score;
        }

        let (rk, bk) = board.find_kings();
        if rk.is_none() {
            return if side == Color::Red { -100000 } else { 100000 };
        }
        if bk.is_none() {
            return if side == Color::Red { 100000 } else { -100000 };
        }

        let phase = game_phase(board);
        let mut score = 0;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.cells[y][x] {
                    let x_usize = x as usize;
                    let y_usize = y as usize;
                    let sign = if piece.color == Color::Red { 1 } else { -1 };
                    let pos = Coord::new(x as i8, y as i8);

                    let mg_v = MG_VALUE[piece.piece_type as usize];
                    let eg_v = EG_VALUE[piece.piece_type as usize];
                    let val = (mg_v as f32 * phase + eg_v as f32 * (1.0 - phase)) as i32;
                    score += val * sign;

                    let pst = pst_val(piece.piece_type, piece.color, x_usize, y_usize, phase);
                    score += pst;

                    match piece.piece_type {
                        PieceType::Horse => {
                            let mob = horse_mobility(board, pos, piece.color);
                            score += mob * sign * 5;
                        }
                        PieceType::Chariot => {
                            let mob = chariot_mobility(board, pos, piece.color);
                            score += mob * sign;
                        }
                        PieceType::Cannon => {
                            let sup = cannon_support(board, pos, piece.color);
                            score += sup * sign;
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(ks_red) = king_safety(board, Color::Red, phase) {
            score += ks_red;
        }
        if let Some(ks_black) = king_safety(board, Color::Black, phase) {
            score -= ks_black;
        }

        score += pawn_structure(board, Color::Red, phase);
        score -= pawn_structure(board, Color::Black, phase);

        score += center_control(board, Color::Red, phase);
        score -= center_control(board, Color::Black, phase);
        score += piece_coordination(board, Color::Red, phase);
        score -= piece_coordination(board, Color::Black, phase);

        score += elephant_structure(board, Color::Red, phase);
        score -= elephant_structure(board, Color::Black, phase);

        if board.is_check(Color::Black) {
            score += 50;
        }
        if board.is_check(Color::Red) {
            score -= 50;
        }

        if side == Color::Red { score } else { -score }
    }
}

pub mod search {
    use super::*;
    use eval::evaluate;
    use movegen::*;
    use super::book::OpeningBook;

    #[derive(Clone)]
    pub struct SharedTT {
        pub tt: Arc<RwLock<TranspositionTable>>,
    }

    impl SharedTT {
        pub fn new() -> Self {
            SharedTT {
                tt: Arc::new(RwLock::new(TranspositionTable::new())),
            }
        }

        #[inline(always)]
        pub fn store(&self, key: u64, depth: u8, value: i32, entry_type: TTEntryType, best_move: Option<Action>) {
            if let Ok(mut tt) = self.tt.write() {
                tt.store(key, depth, value, entry_type, best_move);
            }
        }

        #[inline(always)]
        pub fn probe(&self, key: u64) -> Option<TTEntry> {
            if let Ok(tt) = self.tt.read() {
                tt.probe(key).copied()
            } else {
                None
            }
        }
    }

    #[derive(Clone)]
    pub struct TimeContext {
        pub start_time: Instant,
        pub time_limit: Duration,
        pub stop_flag: Arc<AtomicBool>,
        pub nodes_searched: Arc<AtomicU64>,
    }

    impl TimeContext {
        #[inline(always)]
        pub fn is_time_up(&self) -> bool {
            if self.stop_flag.load(Ordering::Relaxed) {
                return true;
            }
            if self.start_time.elapsed() >= self.time_limit - Duration::from_millis(TIME_BUFFER_MS) {
                self.stop_flag.store(true, Ordering::Relaxed);
                return true;
            }
            false
        }

        #[inline(always)]
        pub fn add_node(&self) {
            self.nodes_searched.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub struct ThreadContext {
        pub history_table: [[i32; 90]; 90],
        pub killer_moves: [[Option<Action>; 2]; (MAX_DEPTH + 4) as usize],
        pub counter_moves: [[Option<Action>; 90]; 90],
    }

    impl ThreadContext {
        pub fn new() -> Self {
            ThreadContext {
                history_table: [[0; 90]; 90],
                killer_moves: [[None; 2]; (MAX_DEPTH + 4) as usize],
                counter_moves: [[None; 90]; 90],
            }
        }

        #[inline(always)]
        pub fn update_history(&mut self, action: Action, depth: u8) {
            let zobrist = get_zobrist();
            let from_idx = zobrist.pos_idx(action.src);
            let to_idx = zobrist.pos_idx(action.tar);
            self.history_table[from_idx][to_idx] += (depth * depth) as i32;
            if self.history_table[from_idx][to_idx] > 2_000_000 {
                for row in &mut self.history_table {
                    for val in row {
                        *val /= 2;
                    }
                }
            }
        }

        #[inline(always)]
        pub fn update_killer(&mut self, action: Action, depth: u8) {
            let depth_idx = depth as usize;
            if depth_idx >= self.killer_moves.len() {
                return;
            }
            if self.killer_moves[depth_idx][0] != Some(action) {
                self.killer_moves[depth_idx][1] = self.killer_moves[depth_idx][0];
                self.killer_moves[depth_idx][0] = Some(action);
            }
        }

        #[inline(always)]
        pub fn update_counter(&mut self, prev_action: Action, current_action: Action) {
            let zobrist = get_zobrist();
            let prev_from = zobrist.pos_idx(prev_action.src);
            let prev_to = zobrist.pos_idx(prev_action.tar);
            self.counter_moves[prev_from][prev_to] = Some(current_action);
        }

        #[inline(always)]
        pub fn age_tables(&mut self) {
            for row in &mut self.history_table {
                for val in row {
                    *val = *val * 3 / 4;
                }
            }
        }

        pub fn sort_moves(
            &self,
            moves: &mut Vec<Action>,
            tt_move: Option<Action>,
            prev_action: Option<Action>,
            depth: u8,
            is_in_check: bool,
            board: &Board,
        ) {
            let depth_idx = depth as usize;
            let zobrist = get_zobrist();

            moves.sort_by(|a, b| {
                let a_is_tt = tt_move == Some(*a);
                let b_is_tt = tt_move == Some(*b);
                if a_is_tt != b_is_tt {
                    return b_is_tt.cmp(&a_is_tt);
                }

                if is_in_check {
                    let a_see = see(board, a.src, a.tar);
                    let b_see = see(board, b.src, b.tar);
                    if a_see != b_see {
                        return b_see.cmp(&a_see);
                    }
                }

                if a.is_check != b.is_check {
                    return b.is_check.cmp(&a.is_check);
                }

                let a_see = see(board, a.src, a.tar);
                let b_see = see(board, b.src, b.tar);
                if a_see != b_see {
                    return b_see.cmp(&a_see);
                }

                let a_mvv = a.mvv_lva_score();
                let b_mvv = b.mvv_lva_score();
                if a_mvv != b_mvv {
                    return b_mvv.cmp(&a_mvv);
                }

                let a_is_counter = prev_action.map_or(false, |pa| {
                    let prev_from = zobrist.pos_idx(pa.src);
                    let prev_to = zobrist.pos_idx(pa.tar);
                    self.counter_moves[prev_from][prev_to] == Some(*a)
                });
                let b_is_counter = prev_action.map_or(false, |pa| {
                    let prev_from = zobrist.pos_idx(pa.src);
                    let prev_to = zobrist.pos_idx(pa.tar);
                    self.counter_moves[prev_from][prev_to] == Some(*b)
                });
                if a_is_counter != b_is_counter {
                    return b_is_counter.cmp(&a_is_counter);
                }

                let a_is_killer = self.killer_moves[depth_idx].contains(&Some(*a));
                let b_is_killer = self.killer_moves[depth_idx].contains(&Some(*b));
                if a_is_killer != b_is_killer {
                    return b_is_killer.cmp(&a_is_killer);
                }

                let a_from = zobrist.pos_idx(a.src);
                let a_to = zobrist.pos_idx(a.tar);
                let b_from = zobrist.pos_idx(b.src);
                let b_to = zobrist.pos_idx(b.tar);
                self.history_table[b_from][b_to].cmp(&self.history_table[a_from][a_to])
            });
        }
    }

    pub fn quiescence(
        board: &mut Board,
        thread_ctx: &mut ThreadContext,
        shared_tt: &SharedTT,
        mut alpha: i32,
        mut beta: i32,
        side: Color,
        depth: u8,
        time_ctx: &TimeContext,
    ) -> i32 {
        time_ctx.add_node();

        if time_ctx.is_time_up() {
            return alpha;
        }

        if depth >= QS_MAX_DEPTH {
            return evaluate(board, side);
        }

        if let Some(winner) = board.is_repetition_violation() {
            return if winner == side { 100000 } else { -100000 };
        }

        let key = board.zobrist_key;
        if let Some(entry) = shared_tt.probe(key) {
            if entry.depth >= depth {
                match entry.entry_type {
                    TTEntryType::Exact => return entry.value,
                    TTEntryType::Lower => alpha = alpha.max(entry.value),
                    TTEntryType::Upper => beta = beta.min(entry.value),
                }
                if alpha >= beta {
                    return entry.value;
                }
            }
        }

        let stand_pat = evaluate(board, side);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let is_in_check = board.is_check(side);
        let mut moves = if is_in_check {
            generate_legal_moves(board, side)
        } else {
            let mut captures = generate_capture_moves(board, side);
            captures.retain(|a| see(board, a.src, a.tar) >= SEE_MARGIN);
            captures
        };

        moves.sort_by(|a, b| see(board, b.src, b.tar).cmp(&see(board, a.src, a.tar)));

        let original_alpha = alpha;
        let mut best_eval = stand_pat;

        for action in moves {
            if !is_in_check && action.captured.is_some() {
                let capture_value = see(board, action.src, action.tar);
                if stand_pat + capture_value + 200 < alpha {
                    continue;
                }
            }

            board.make_move(action);
            let eval = -quiescence(
                board, thread_ctx, shared_tt, -beta, -alpha,
                side.opponent(), depth + 1, time_ctx
            );
            board.undo_move(action);

            if time_ctx.is_time_up() {
                return alpha;
            }

            if eval > best_eval {
                best_eval = eval;
            }
            if eval > alpha {
                alpha = eval;
            }
            if alpha >= beta {
                break;
            }
        }

        let entry_type = if best_eval <= original_alpha {
            TTEntryType::Upper
        } else if best_eval >= beta {
            TTEntryType::Lower
        } else {
            TTEntryType::Exact
        };
        shared_tt.store(key, depth, best_eval, entry_type, None);

        best_eval
    }

    pub fn zw_search(
        board: &mut Board,
        thread_ctx: &mut ThreadContext,
        shared_tt: &SharedTT,
        depth: u8,
        beta: i32,
        side: Color,
        pv_node: bool,
        best_action: &mut Option<Action>,
        time_ctx: &TimeContext,
        extension_count: u8,
        prev_action: Option<Action>,
    ) -> i32 {
        time_ctx.add_node();
        let mut alpha = beta - 1;
        let original_alpha = alpha;
        let key = board.zobrist_key;
        let is_in_check = board.is_check(side);

        if time_ctx.is_time_up() {
            return alpha;
        }

        if let Some(winner) = board.is_repetition_violation() {
            return if winner == side { 100000 } else { -100000 };
        }

        if let Some(entry) = shared_tt.probe(key) {
            if entry.depth >= depth {
                match entry.entry_type {
                    TTEntryType::Exact => {
                        *best_action = entry.best_move;
                        return entry.value;
                    }
                    TTEntryType::Lower => {
                        if entry.value >= beta {
                            return entry.value;
                        }
                    }
                    TTEntryType::Upper => {
                        if entry.value <= alpha {
                            return entry.value;
                        }
                    }
                }
            }
        }

        if depth == 0 {
            return quiescence(board, thread_ctx, shared_tt, alpha, beta, side, 0, time_ctx);
        }

        let mut moves = generate_legal_moves(board, side);
        if moves.is_empty() {
            return if is_in_check {
                -100000 + (MAX_DEPTH - depth) as i32
            } else {
                0
            };
        }

        let is_endgame = evaluate(board, side).abs() < ENDGAME_THRESHOLD;
        if !pv_node && !is_in_check && !is_endgame && depth >= NULL_MOVE_REDUCTION + 1 {
            let zobrist = get_zobrist();
            board.zobrist_key ^= zobrist.side;
            board.current_side = board.current_side.opponent();

            let null_depth = depth - 1 - NULL_MOVE_REDUCTION;
            let null_eval = -zw_search(
                board, thread_ctx, shared_tt, null_depth, -alpha,
                side.opponent(), false, &mut None, time_ctx, extension_count, None
            );

            board.zobrist_key ^= zobrist.side;
            board.current_side = board.current_side.opponent();

            if null_eval >= beta && !time_ctx.is_time_up() {
                return beta;
            }
        }

        if !pv_node && !is_in_check && depth <= 3 {
            let static_eval = evaluate(board, side);
            let futility_margin = FUTILITY_MARGIN * depth as i32;
            if static_eval + futility_margin <= alpha {
                return static_eval;
            }
        }

        let tt_move = shared_tt.probe(key).and_then(|e| e.best_move);
        thread_ctx.sort_moves(&mut moves, tt_move, prev_action, depth, is_in_check, board);

        let mut best_eval = -i32::MAX;
        let mut current_best_move = None;
        let mut has_pv = false;

        for (move_idx, action) in moves.iter().enumerate() {
            let gives_check = action.is_check;
            let mut extension = 0;
            if (gives_check || is_in_check) && extension_count < MAX_CHECK_EXTENSION {
                extension = 1;
            }
            let new_extension_count = if gives_check || is_in_check {
                extension_count + 1
            } else {
                extension_count
            };
            if new_extension_count > MAX_TOTAL_EXTENSION {
                extension = 0;
            }
            let new_depth = depth - 1 + extension;

            board.make_move(*action);

            let mut eval;
            if has_pv && !pv_node && !is_in_check && !gives_check && action.captured.is_none() && move_idx >= LMR_MIN_MOVES && depth >= 3 {
                let reduced_depth = depth - 2;
                eval = -zw_search(
                    board, thread_ctx, shared_tt, reduced_depth, -alpha,
                    side.opponent(), false, &mut None, time_ctx, new_extension_count, Some(*action)
                );
                if eval > alpha && !time_ctx.is_time_up() {
                    eval = -zw_search(
                        board, thread_ctx, shared_tt, new_depth, -alpha,
                        side.opponent(), true, &mut None, time_ctx, new_extension_count, Some(*action)
                    );
                }
            } else {
                eval = -zw_search(
                    board, thread_ctx, shared_tt, new_depth, -alpha,
                    side.opponent(), !has_pv && pv_node, &mut None, time_ctx, new_extension_count, Some(*action)
                );
            }

            board.undo_move(*action);

            if time_ctx.is_time_up() {
                return best_eval.max(alpha);
            }

            if eval > best_eval {
                best_eval = eval;
                current_best_move = Some(*action);
                *best_action = Some(*action);
            }

            if eval >= beta {
                if action.captured.is_none() {
                    thread_ctx.update_killer(*action, depth);
                    thread_ctx.update_history(*action, depth);
                    if let Some(pa) = prev_action {
                        thread_ctx.update_counter(pa, *action);
                    }
                }
                break;
            }

            if eval > alpha {
                alpha = eval;
                has_pv = true;
            }
        }

        let entry_type = if best_eval <= original_alpha {
            TTEntryType::Upper
        } else if best_eval >= beta {
            TTEntryType::Lower
        } else {
            TTEntryType::Exact
        };
        shared_tt.store(key, depth, best_eval, entry_type, current_best_move);

        best_eval
    }

    pub fn mtdf(
        board: &mut Board,
        thread_ctx: &mut ThreadContext,
        shared_tt: &SharedTT,
        max_depth: u8,
        first_guess: i32,
        side: Color,
        best_action: &mut Option<Action>,
        time_ctx: &TimeContext,
    ) -> i32 {
        let mut lower_bound = -100000;
        let mut upper_bound = 100000;
        let mut guess = first_guess;
        let mut current_best = None;

        while lower_bound < upper_bound && !time_ctx.is_time_up() {
            let beta = if guess == lower_bound { guess + 1 } else { guess };
            let mut temp_best = None;

            let value = zw_search(
                board, thread_ctx, shared_tt, max_depth, beta,
                side, true, &mut temp_best, time_ctx, 0, None
            );

            if value < beta {
                upper_bound = value;
            } else {
                lower_bound = value;
            }
            guess = value;

            if temp_best.is_some() {
                current_best = temp_best;
            }
        }

        *best_action = current_best;
        guess
    }

    pub fn search_with_aspiration(
        board: &mut Board,
        thread_ctx: &mut ThreadContext,
        shared_tt: &SharedTT,
        depth: u8,
        prev_score: i32,
        side: Color,
        best_action: &mut Option<Action>,
        time_ctx: &TimeContext,
    ) -> i32 {
        if depth <= 1 || prev_score == 0 {
            return mtdf(board, thread_ctx, shared_tt, depth, 0, side, best_action, time_ctx);
        }

        let mut alpha = -100000;
        let mut beta = 100000;
        let mut score = prev_score;
        let mut window = ASPIRATION_WINDOW;

        loop {
            let mut current_best = None;
            let low = (score - window).max(alpha);
            let high = (score + window).min(beta);

            let mut mtdf_guess = score;
            let mut mtdf_lower = low;
            let mut mtdf_upper = high;

            while mtdf_lower < mtdf_upper && !time_ctx.is_time_up() {
                let beta_val = if mtdf_guess == mtdf_lower { mtdf_guess + 1 } else { mtdf_guess };
                let mut temp_best = None;

                let value = zw_search(
                    board, thread_ctx, shared_tt, depth, beta_val,
                    side, true, &mut temp_best, time_ctx, 0, None
                );

                if value < beta_val {
                    mtdf_upper = value;
                } else {
                    mtdf_lower = value;
                }
                mtdf_guess = value;

                if temp_best.is_some() {
                    current_best = temp_best;
                }
            }

            score = mtdf_guess;

            if score <= low {
                window *= 2;
                beta = low;
            } else if score >= high {
                window *= 2;
                alpha = high;
            } else {
                *best_action = current_best;
                return score;
            }

            if window > 1000 {
                return mtdf(board, thread_ctx, shared_tt, depth, 0, side, best_action, time_ctx);
            }
        }
    }

    fn worker_thread(
        mut board: Board,
        shared_tt: SharedTT,
        time_ctx: TimeContext,
        thread_id: usize,
        result_sender: std::sync::mpsc::Sender<(u8, i32, Option<Action>)>,
    ) {
        let mut thread_ctx = ThreadContext::new();
        let mut last_guess = 0;
        let side = board.current_side;

        let start_depth = if thread_id == 0 { 1 } else { thread_id as u8 % 3 + 1 };

        for depth in start_depth..=MAX_DEPTH {
            if time_ctx.is_time_up() {
                break;
            }

            let mut current_best = None;
            let score = search_with_aspiration(
                &mut board, &mut thread_ctx, &shared_tt,
                depth, last_guess, side, &mut current_best, &time_ctx
            );

            if !time_ctx.is_time_up() {
                last_guess = score;
                thread_ctx.age_tables();
                let _ = result_sender.send((depth, score, current_best));
            } else {
                break;
            }
        }
    }

    pub fn find_best_move(board: &mut Board, _max_depth: u8, side: Color) -> Option<Action> {
        if board.move_history.len() < 15 {
            static BOOK: OnceLock<OpeningBook> = OnceLock::new();
            let book = BOOK.get_or_init(OpeningBook::new);
            if let Some(action) = book.probe(board) {
                let legal_moves = generate_legal_moves(board, side);
                if legal_moves.contains(&action) {
                    return Some(action);
                }
            }
        }

        let legal_moves = generate_legal_moves(board, side);
        if legal_moves.is_empty() {
            return None;
        }

        let shared_tt = SharedTT::new();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let nodes_searched = Arc::new(AtomicU64::new(0));
        let time_ctx = TimeContext {
            start_time: Instant::now(),
            time_limit: Duration::from_millis(SEARCH_TIMEOUT_MS),
            stop_flag: Arc::clone(&stop_flag),
            nodes_searched: Arc::clone(&nodes_searched),
        };

        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        let mut handles = Vec::with_capacity(SEARCH_THREADS);

        for thread_id in 0..SEARCH_THREADS {
            let board_clone = board.clone();
            let tt_clone = shared_tt.clone();
            let time_ctx_clone = time_ctx.clone();
            let sender_clone = result_sender.clone();

            let handle = thread::spawn(move || {
                worker_thread(board_clone, tt_clone, time_ctx_clone, thread_id, sender_clone);
            });
            handles.push(handle);
        }

        let stop_flag_clone = Arc::clone(&stop_flag);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(SEARCH_TIMEOUT_MS));
            stop_flag_clone.store(true, Ordering::Relaxed);
        });

        let mut best_depth = 0;
        let mut best_score = -100000;
        let mut final_best_action = legal_moves.first().copied();

        while !time_ctx.is_time_up() {
            match result_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok((depth, score, action)) => {
                    if depth > best_depth || (depth == best_depth && score > best_score) {
                        best_depth = depth;
                        best_score = score;
                        if let Some(m) = action {
                            final_best_action = Some(m);
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            }
        }

        stop_flag.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.join();
        }

        final_best_action
    }
}

fn print_board(board: &Board, order: u8) {
    match order {
        1 => println!("{}", board),
        2 => {
            let mut board_clone = board.clone();
            board_clone.flip_vertically();
            println!("{}", board_clone);
        }
        _ => unreachable!(),
    }
}

fn parse_coord(s: &str, order: u8, is_y: bool) -> Result<i8, &'static str> {
    let val = s.parse::<i8>().map_err(|_| "无效坐标")?;
    if val < 0 || (is_y && val > 9) || (!is_y && val > 8) {
        return Err("坐标超出范围");
    }
    let adjusted = if is_y && order == 2 { 9 - val } else { val };
    Ok(adjusted)
}

fn ai_move(board: &mut Board, order: u8, depth: u8) -> bool {
    if let Some(action) = search::find_best_move(board, depth, Color::Red) {
        println!(
            "AI走棋: ({}, {}) -> ({}, {})",
            action.src.x,
            if order == 1 { action.src.y } else { 9 - action.src.y },
            action.tar.x,
            if order == 1 { action.tar.y } else { 9 - action.tar.y }
        );
        board.make_move(action);
        print_board(board, order);
        true
    } else {
        println!("AI无子可走，你赢了！");
        false
    }
}

fn check_game_over(board: &Board) -> bool {
    if let Some(winner) = board.get_winner() {
        match winner {
            Color::Black => println!("游戏结束：你赢了！"),
            Color::Red => println!("游戏结束：你输了！"),
        }
        return true;
    }

    if let Some(winner) = board.is_repetition_violation() {
        match winner {
            Color::Black => println!("游戏结束：你赢了，AI违规（长将/长捉）"),
            Color::Red => println!("游戏结束：你输了，违规（长将/长捉）"),
        }
        return true;
    }

    let has_legal_move = |color: Color| {
        let mut board_clone = board.clone();
        !movegen::generate_legal_moves(&mut board_clone, color).is_empty()
    };
    let red_has_move = has_legal_move(Color::Red);
    let black_has_move = has_legal_move(Color::Black);

    if !red_has_move && !black_has_move {
        println!("游戏结束：平局");
        true
    } else if !red_has_move {
        println!("游戏结束：你赢了，AI无子可动");
        true
    } else if !black_has_move {
        println!("游戏结束：你输了，无子可动");
        true
    } else {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 中国象棋引擎 ===");
    let stdin = io::stdin();
    let mut input = String::new();

    println!("\n【规则选择】");
    println!("1. {}", RuleSet::Official.description());
    println!("2. {}", RuleSet::OnlyLongCheckIllegal.description());
    println!("3. {}", RuleSet::NoRestriction.description());
    print!("请输入规则编号（1-3，默认1）：");
    io::stdout().flush()?;

    input.clear();
    stdin.read_line(&mut input)?;
    let rule_set = match input.trim().parse::<u8>().unwrap_or(1) {
        1 => RuleSet::Official,
        2 => RuleSet::OnlyLongCheckIllegal,
        3 => RuleSet::NoRestriction,
        _ => RuleSet::Official,
    };
    println!("已选择规则：{}", rule_set);

    println!("\n【先后手选择】");
    println!("1. AI先手（红方）");
    println!("2. 玩家先手（黑方）");
    print!("请输入顺序编号（1-2，默认2）：");
    io::stdout().flush()?;

    input.clear();
    stdin.read_line(&mut input)?;
    let order = input.trim().parse::<u8>().unwrap_or(2);
    if !(1..=2).contains(&order) {
        return Err("无效的顺序编号".into());
    }
    println!("已选择：{}", if order == 1 { "AI先手" } else { "玩家先手" });

    let mut board = Board::new(rule_set, order);
    println!("\n=== 游戏开始 ===");
    print_board(&board, order);

    let mut ai_turn = order == 1;

    loop {
        if ai_turn {
            if !ai_move(&mut board, order, MAX_DEPTH) {
                break;
            }
            if check_game_over(&board) {
                break;
            }
        }

        'input_loop: loop {
            println!("\n请输入你的走法 (格式: x1 y1 x2 y2)：");
            input.clear();
            stdin.read_line(&mut input)?;
            let parts: Vec<&str> = input.trim().split_whitespace().collect();

            if parts.len() != 4 {
                println!("输入格式错误，请重新输入！");
                continue 'input_loop;
            }

            let x1 = match parse_coord(parts[0], order, false) {
                Ok(v) => v,
                Err(e) => {
                    println!("{}", e);
                    continue 'input_loop;
                }
            };
            let y1 = match parse_coord(parts[1], order, true) {
                Ok(v) => v,
                Err(e) => {
                    println!("{}", e);
                    continue 'input_loop;
                }
            };
            let x2 = match parse_coord(parts[2], order, false) {
                Ok(v) => v,
                Err(e) => {
                    println!("{}", e);
                    continue 'input_loop;
                }
            };
            let y2 = match parse_coord(parts[3], order, true) {
                Ok(v) => v,
                Err(e) => {
                    println!("{}", e);
                    continue 'input_loop;
                }
            };

            let src = Coord::new(x1, y1);
            let tar = Coord::new(x2, y2);

            match board.get(src) {
                Some(p) if p.color == Color::Black => {}
                _ => {
                    println!("该位置没有你的棋子！");
                    continue 'input_loop;
                }
            };

            let mut board_clone = board.clone();
            let legal_moves = movegen::generate_legal_moves(&mut board_clone, Color::Black);
            if !legal_moves.iter().any(|m| m.src == src && m.tar == tar) {
                println!("非法走法！");
                continue 'input_loop;
            }

            let captured = board.get(tar);
            let action = Action::new(src, tar, captured);
            board.make_move(action);
            println!("走棋成功！");
            print_board(&board, order);
            break 'input_loop;
        }

        if check_game_over(&board) {
            break;
        }

        ai_turn = true;
    }

    Ok(())
}
