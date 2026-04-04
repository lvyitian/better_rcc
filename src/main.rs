use std::fmt;
use std::io;
use std::io::Write;
use std::sync::OnceLock;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

pub const BOARD_WIDTH: i8 = 9;
pub const BOARD_HEIGHT: i8 = 10;
pub const MAX_DEPTH: u8 = 8;          // 改回 8 层深度
pub const QS_MAX_DEPTH: u8 = 8;
pub const MAX_CHECK_EXTENSION: u8 = 2;
pub const TT_SIZE: usize = 1 << 24;
pub const ENDGAME_THRESHOLD: i32 = 2000;
pub const MIDGAME_THRESHOLD: i32 = 4000;
pub const REPETITION_VIOLATION_COUNT: u8 = 3;
pub const SEARCH_TIMEOUT_MS: u64 = 15000; // 改为 15 秒超时

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
            Color::Black => self.y <= 2,
            Color::Red => self.y >= 7,
        };
        x_ok && y_ok
    }

    #[inline(always)]
    pub fn crosses_river(self, color: Color) -> bool {
        match color {
            Color::Black => self.y >= 5,
            Color::Red => self.y <= 4,
        }
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

    pub fn generate_legal_moves(board: &mut Board, color: Color) -> Vec<Action> {
        let mut legal_moves = Vec::with_capacity(40);
        let pseudo_moves = generate_pseudo_moves(board, color);

        for mut action in pseudo_moves {
            let src = action.src;
            let tar = action.tar;
            let piece = board.get(src).unwrap();
            let captured = action.captured;

            board.cells[tar.y as usize][tar.x as usize] = Some(piece);
            board.cells[src.y as usize][src.x as usize] = None;

            let is_self_checked = board.is_check(color);
            let legal = !is_self_checked;

            action.is_check = board.is_check(color.opponent());
            action.is_capture_threat = board.is_capture_threat_internal(color);

            board.cells[src.y as usize][src.x as usize] = Some(piece);
            board.cells[tar.y as usize][tar.x as usize] = captured;

            if legal {
                legal_moves.push(action);
            }
        }

        legal_moves
    }

    pub fn generate_capture_moves(board: &mut Board, color: Color) -> Vec<Action> {
        let mut moves = generate_pseudo_moves(board, color);
        moves.retain(|a| a.captured.is_some());

        let mut legal_captures = Vec::new();
        for action in moves {
            let src = action.src;
            let tar = action.tar;
            let piece = board.get(src).unwrap();
            let captured = action.captured;

            board.cells[tar.y as usize][tar.x as usize] = Some(piece);
            board.cells[src.y as usize][src.x as usize] = None;

            let legal = !board.is_check(color);

            board.cells[src.y as usize][src.x as usize] = Some(piece);
            board.cells[tar.y as usize][tar.x as usize] = captured;

            if legal {
                legal_captures.push(action);
            }
        }

        legal_captures
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
    pub fn new(rule_set: RuleSet) -> Self {
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
            current_side: Color::Red,
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
        let piece = self.get(action.src).unwrap();
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

    const BASE_VALUE_MIDGAME: [i32; 7] = [10000, 110, 110, 70, 320, 320, 600];
    const BASE_VALUE_ENDGAME: [i32; 7] = [10000, 110, 110, 150, 300, 200, 650];

    const PST_KING_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 2, 3, 2, 0, 0, 0],
        [0, 0, 0, 10, 15, 10, 0, 0, 0],
    ];

    const PST_ADVISOR_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 20, 0, 20, 0, 0, 0],
        [0, 0, 0, 0, 20, 0, 0, 0, 0],
    ];

    const PST_ELEPHANT_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 20, 0, 0, 0, 20, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [20, 0, 0, 0, 20, 0, 0, 0, 20],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const PST_HORSE_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 30, 30, 30, 30, 30, 20, 10],
        [20, 30, 50, 50, 50, 50, 50, 30, 20],
        [10, 20, 30, 40, 40, 40, 30, 20, 10],
        [10, 10, 20, 20, 20, 20, 20, 10, 10],
    ];

    const PST_CHARIOT_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [20, 20, 20, 20, 20, 20, 20, 20, 20],
        [30, 40, 40, 40, 40, 40, 40, 40, 30],
        [10, 10, 20, 30, 30, 30, 20, 10, 10],
        [10, 20, 20, 30, 30, 30, 20, 20, 10],
    ];

    const PST_CANNON_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 10, 10, 20, 20, 20, 10, 10, 10],
        [10, 20, 20, 20, 20, 20, 20, 20, 10],
        [5, 10, 10, 15, 15, 15, 10, 10, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const PST_PAWN_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [90, 90, 90, 110, 130, 110, 90, 90, 90],
        [70, 70, 70, 90, 110, 90, 70, 70, 70],
        [50, 50, 50, 70, 90, 70, 50, 50, 50],
        [30, 30, 30, 30, 30, 30, 30, 30, 30],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    fn game_phase(board: &Board) -> f32 {
        let mut total_material = 0;
        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = board.cells[y][x] {
                    if p.piece_type != PieceType::King {
                        total_material += BASE_VALUE_MIDGAME[p.piece_type as usize];
                    }
                }
            }
        }
        (total_material - ENDGAME_THRESHOLD).max(0) as f32 / (MIDGAME_THRESHOLD - ENDGAME_THRESHOLD) as f32
    }

    fn position_value(pt: PieceType, color: Color, x: i8, y: i8) -> i32 {
        let (x_usize, y_usize) = (x as usize, y as usize);
        let y_mirrored = 9 - y_usize;

        let val = match pt {
            PieceType::King =>
                if color == Color::Red { PST_KING_RED[y_usize][x_usize] } else { PST_KING_RED[y_mirrored][x_usize] }
            PieceType::Advisor =>
                if color == Color::Red { PST_ADVISOR_RED[y_usize][x_usize] } else { PST_ADVISOR_RED[y_mirrored][x_usize] }
            PieceType::Elephant =>
                if color == Color::Red { PST_ELEPHANT_RED[y_usize][x_usize] } else { PST_ELEPHANT_RED[y_mirrored][x_usize] }
            PieceType::Horse =>
                if color == Color::Red { PST_HORSE_RED[y_usize][x_usize] } else { PST_HORSE_RED[y_mirrored][x_usize] }
            PieceType::Chariot =>
                if color == Color::Red { PST_CHARIOT_RED[y_usize][x_usize] } else { PST_CHARIOT_RED[y_mirrored][x_usize] }
            PieceType::Cannon =>
                if color == Color::Red { PST_CANNON_RED[y_usize][x_usize] } else { PST_CANNON_RED[y_mirrored][x_usize] }
            PieceType::Pawn =>
                if color == Color::Red { PST_PAWN_RED[y_usize][x_usize] } else { PST_PAWN_RED[y_mirrored][x_usize] }
        };

        val * color.sign()
    }

    pub fn evaluate(board: &Board, side: Color) -> i32 {
        let phase = game_phase(board).clamp(0.0, 1.0);
        let mut score = 0;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.cells[y][x] {
                    let base_mid = BASE_VALUE_MIDGAME[piece.piece_type as usize];
                    let base_end = BASE_VALUE_ENDGAME[piece.piece_type as usize];
                    let base = (base_mid as f32 * phase + base_end as f32 * (1.0 - phase)) as i32;
                    let pos = position_value(piece.piece_type, piece.color, x as i8, y as i8);

                    if piece.color == Color::Red {
                        score += base + pos;
                    } else {
                        score -= base + pos;
                    }
                }
            }
        }

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
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    pub struct SearchContext {
        pub tt: TranspositionTable,
        pub history_table: [[i32; 90]; 90],
        pub killer_moves: [[Option<Action>; 2]; (MAX_DEPTH + 2) as usize],
        pub nodes_searched: u64,
    }

    impl SearchContext {
        pub fn new() -> Self {
            SearchContext {
                tt: TranspositionTable::new(),
                history_table: [[0; 90]; 90],
                killer_moves: [[None; 2]; (MAX_DEPTH + 2) as usize],
                nodes_searched: 0,
            }
        }

        #[inline(always)]
        pub fn update_history(&mut self, action: Action, depth: u8) {
            let zobrist = get_zobrist();
            let from_idx = zobrist.pos_idx(action.src);
            let to_idx = zobrist.pos_idx(action.tar);
            self.history_table[from_idx][to_idx] += (depth * depth) as i32;
            if self.history_table[from_idx][to_idx] > 1_000_000 {
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
            if depth_idx >= (MAX_DEPTH + 2) as usize {
                return;
            }
            if self.killer_moves[depth_idx][0] != Some(action) {
                self.killer_moves[depth_idx][1] = self.killer_moves[depth_idx][0];
                self.killer_moves[depth_idx][0] = Some(action);
            }
        }

        pub fn sort_moves(&self, moves: &mut Vec<Action>, tt_move: Option<Action>, depth: u8) {
            let depth_idx = depth as usize;
            let zobrist = get_zobrist();

            moves.sort_by(|a, b| {
                let a_is_tt = tt_move == Some(*a);
                let b_is_tt = tt_move == Some(*b);
                if a_is_tt != b_is_tt {
                    return b_is_tt.cmp(&a_is_tt);
                }

                if a.is_check != b.is_check {
                    return b.is_check.cmp(&a.is_check);
                }

                let a_mvv = a.mvv_lva_score();
                let b_mvv = b.mvv_lva_score();
                if a_mvv != b_mvv {
                    return b_mvv.cmp(&a_mvv);
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
        ctx: &mut SearchContext,
        mut alpha: i32,
        beta: i32,
        side: Color,
        depth: u8,
        stop_flag: &AtomicBool,
    ) -> i32 {
        ctx.nodes_searched += 1;

        if stop_flag.load(Ordering::Relaxed) || depth >= QS_MAX_DEPTH {
            return evaluate(board, side);
        }

        if let Some(winner) = board.is_repetition_violation() {
            return if winner == side { 100000 } else { -100000 };
        }

        let stand_pat = evaluate(board, side);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut moves = generate_capture_moves(board, side);
        moves.sort_by(|a, b| b.mvv_lva_score().cmp(&a.mvv_lva_score()));

        for action in moves {
            board.make_move(action);
            let eval = -quiescence(board, ctx, -beta, -alpha, side.opponent(), depth + 1, stop_flag);
            board.undo_move(action);

            if stop_flag.load(Ordering::Relaxed) {
                return alpha;
            }

            if eval >= beta {
                return beta;
            }
            if eval > alpha {
                alpha = eval;
            }
        }

        alpha
    }

    pub fn pvs(
        board: &mut Board,
        ctx: &mut SearchContext,
        depth: u8,
        mut alpha: i32,
        mut beta: i32,
        side: Color,
        pv_node: bool,
        best_action: &mut Option<Action>,
        stop_flag: &AtomicBool,
        extension_count: u8,
    ) -> i32 {
        ctx.nodes_searched += 1;
        let original_alpha = alpha;
        let key = board.zobrist_key;

        if stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        if let Some(winner) = board.is_repetition_violation() {
            return if winner == side { 100000 } else { -100000 };
        }

        if let Some(entry) = ctx.tt.probe(key) {
            if entry.depth >= depth {
                match entry.entry_type {
                    TTEntryType::Exact => {
                        *best_action = entry.best_move;
                        return entry.value;
                    }
                    TTEntryType::Lower => alpha = alpha.max(entry.value),
                    TTEntryType::Upper => beta = beta.min(entry.value),
                }
                if alpha >= beta {
                    return entry.value;
                }
            }
        }

        if depth == 0 {
            return quiescence(board, ctx, alpha, beta, side, 0, stop_flag);
        }

        let mut moves = generate_legal_moves(board, side);
        if moves.is_empty() {
            return if board.is_check(side) { -100000 + (MAX_DEPTH - depth) as i32 } else { 0 };
        }

        let tt_move = ctx.tt.probe(key).and_then(|e| e.best_move);
        ctx.sort_moves(&mut moves, tt_move, depth);

        let mut best_eval = -i32::MAX;
        let mut current_best_move = None;
        let mut has_pv = false;

        for (_i, action) in moves.iter().enumerate() {
            let extension = if action.is_check && extension_count < MAX_CHECK_EXTENSION { 1 } else { 0 };
            let new_extension_count = if action.is_check { extension_count + 1 } else { extension_count };

            board.make_move(*action);

            let mut eval;

            if has_pv {
                eval = -pvs(board, ctx, depth - 1 + extension, -alpha - 1, -alpha, side.opponent(), false, &mut None, stop_flag, new_extension_count);
                if eval > alpha && eval < beta && !stop_flag.load(Ordering::Relaxed) {
                    eval = -pvs(board, ctx, depth - 1 + extension, -beta, -alpha, side.opponent(), true, &mut None, stop_flag, new_extension_count);
                }
            } else {
                eval = -pvs(board, ctx, depth - 1 + extension, -beta, -alpha, side.opponent(), pv_node, &mut None, stop_flag, new_extension_count);
            }

            board.undo_move(*action);

            if stop_flag.load(Ordering::Relaxed) {
                return best_eval.max(alpha);
            }

            if eval > best_eval {
                best_eval = eval;
                current_best_move = Some(*action);
                *best_action = Some(*action);
            }

            if eval > alpha {
                alpha = eval;
                has_pv = true;
            }

            if alpha >= beta {
                if action.captured.is_none() {
                    ctx.update_killer(*action, depth);
                    ctx.update_history(*action, depth);
                }
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
        ctx.tt.store(key, depth, best_eval, entry_type, current_best_move);

        best_eval
    }

    pub fn find_best_move(board: &mut Board, max_depth: u8, side: Color) -> Option<Action> {
        let legal_moves = generate_legal_moves(board, side);
        if legal_moves.is_empty() {
            return None;
        }

        let mut ctx = SearchContext::new();
        let mut best_action = None;
        let stop_flag = Arc::new(AtomicBool::new(false));

        let stop_flag_clone = Arc::clone(&stop_flag);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(SEARCH_TIMEOUT_MS));
            stop_flag_clone.store(true, Ordering::Relaxed);
        });

        for depth in 1..=max_depth {
            let mut current_best = None;
            let _ = pvs(board, &mut ctx, depth, -100000, 100000, side, true, &mut current_best, &stop_flag, 0);

            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            if let Some(_m) = current_best {
                best_action = current_best;
            }
        }

        if best_action.is_none() {
            best_action = legal_moves.first().copied();
        }

        best_action
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

    let mut board = Board::new(rule_set);
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
