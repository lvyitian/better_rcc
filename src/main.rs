use std::fmt;

// 棋子类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceType {
    King,     // 将/帅
    Advisor,  // 士/仕
    Elephant, // 象/相
    Horse,    // 马
    Chariot,  // 车
    Cannon,   // 炮
    Pawn,     // 卒/兵
}

// 阵营
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Red,
    Black,
}

impl Color {
    fn opponent(self) -> Self {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }
}

// 棋子
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    color: Color,
    piece_type: PieceType,
}

// 坐标
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Coord {
    x: u8,
    y: u8,
}

impl Coord {
    fn new(x: u8, y: u8) -> Self {
        Coord { x, y }
    }

    fn is_valid(self) -> bool {
        self.x < 9 && self.y < 10
    }

    fn in_palace(self, color: Color) -> bool {
        let x_ok = self.x >= 3 && self.x <= 5;
        let y_ok = match color {
            Color::Red => self.y <= 2,
            Color::Black => self.y >= 7,
        };
        x_ok && y_ok
    }

    fn crosses_river(self, color: Color) -> bool {
        match color {
            Color::Red => self.y >= 5,
            Color::Black => self.y <= 4,
        }
    }
}

// 走法
#[derive(Debug, Clone, Copy)]
struct Action {
    src: Coord,
    tar: Coord,
    captured: Option<Piece>,
}

impl Action {
    fn new(src: Coord, tar: Coord, captured: Option<Piece>) -> Self {
        Action { src, tar, captured }
    }
}

// 棋盘
struct Board {
    cells: [[Option<Piece>; 9]; 10],
}

impl Board {
    fn new() -> Self {
        let mut cells = [[None; 9]; 10];

        // 初始化黑方 (y=0-4)
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
        for (x, &pt) in back_row.iter().enumerate() {
            cells[0][x] = Some(Piece {
                color: Color::Black,
                piece_type: pt,
            });
        }
        cells[2][1] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Cannon,
        });
        cells[2][7] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Cannon,
        });
        for x in [0, 2, 4, 6, 8].iter() {
            cells[3][*x] = Some(Piece {
                color: Color::Black,
                piece_type: PieceType::Pawn,
            });
        }

        // 初始化红方 (y=5-9)
        for (x, &pt) in back_row.iter().enumerate() {
            cells[9][x] = Some(Piece {
                color: Color::Red,
                piece_type: pt,
            });
        }
        cells[7][1] = Some(Piece {
            color: Color::Red,
            piece_type: PieceType::Cannon,
        });
        cells[7][7] = Some(Piece {
            color: Color::Red,
            piece_type: PieceType::Cannon,
        });
        for x in [0, 2, 4, 6, 8].iter() {
            cells[6][*x] = Some(Piece {
                color: Color::Red,
                piece_type: PieceType::Pawn,
            });
        }

        Board { cells }
    }

    fn get(&self, coord: Coord) -> Option<Piece> {
        self.cells[coord.y as usize][coord.x as usize]
    }

    fn set(&mut self, coord: Coord, piece: Option<Piece>) {
        self.cells[coord.y as usize][coord.x as usize] = piece;
    }

    fn make_move(&mut self, action: Action) {
        let piece = self.get(action.src).unwrap();
        self.set(action.tar, Some(piece));
        self.set(action.src, None);
    }

    fn undo_move(&mut self, action: Action) {
        let piece = self.get(action.tar).unwrap();
        self.set(action.src, Some(piece));
        self.set(action.tar, action.captured);
    }

    fn is_face_to_face(&self) -> bool {
        let mut red_king = None;
        let mut black_king = None;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(p) = self.cells[y][x] {
                    if p.piece_type == PieceType::King {
                        match p.color {
                            Color::Red => red_king = Some(Coord::new(x as u8, y as u8)),
                            Color::Black => black_king = Some(Coord::new(x as u8, y as u8)),
                        }
                    }
                }
            }
        }

        let (rk, bk) = match (red_king, black_king) {
            (Some(r), Some(b)) => (r, b),
            _ => return false,
        };

        if rk.x != bk.x {
            return false;
        }

        for y in (bk.y + 1)..rk.y {
            if self.cells[y as usize][rk.x as usize].is_some() {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "  0 1 2 3 4 5 6 7 8")?;
        writeln!(f, "  -------------------")?;
        for y in 0..10 {
            write!(f, "{}|", y)?;
            for x in 0..9 {
                let c = match self.cells[y][x] {
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::King,
                    }) => "帥",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Advisor,
                    }) => "仕",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Elephant,
                    }) => "相",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Horse,
                    }) => "馬",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Chariot,
                    }) => "車",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Cannon,
                    }) => "炮",
                    Some(Piece {
                        color: Color::Red,
                        piece_type: PieceType::Pawn,
                    }) => "兵",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::King,
                    }) => "將",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Advisor,
                    }) => "士",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Elephant,
                    }) => "象",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Horse,
                    }) => "馬",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Chariot,
                    }) => "車",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Cannon,
                    }) => "砲",
                    Some(Piece {
                        color: Color::Black,
                        piece_type: PieceType::Pawn,
                    }) => "卒",
                    None => "·",
                };
                write!(f, "{} ", c)?;
            }
            writeln!(f, "|{}", y)?;
        }
        writeln!(f, "  -------------------")?;
        writeln!(f, "  0 1 2 3 4 5 6 7 8")
    }
}

// 评估模块
mod eval {
    use super::*;

    // 棋子基础价值 (单位: 分)
    const fn base_value(pt: PieceType) -> i32 {
        match pt {
            PieceType::King => 10000,
            PieceType::Advisor => 100,
            PieceType::Elephant => 100,
            PieceType::Horse => 300,
            PieceType::Chariot => 500,
            PieceType::Cannon => 300,
            PieceType::Pawn => 50,
        }
    }

    // 红方位置价值表 (y从0到9, 0是黑方底线)
    // 实际上红方在y=5-9，所以我们定义一个通用的表，黑方翻转y坐标
    const PST_PAWN_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 9, 9, 11, 13, 11, 9, 9, 9],
        [19, 24, 34, 42, 44, 42, 34, 24, 19],
        [19, 24, 34, 42, 44, 42, 34, 24, 19],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const PST_HORSE_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    const PST_CHARIOT_RED: [[i32; 9]; 10] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    fn position_value(pt: PieceType, c: Color, x: u8, y: u8) -> i32 {
        let (x_usize, y_usize) = (x as usize, y as usize);
        let y_mirrored = 9 - y_usize;

        let val = match pt {
            PieceType::Pawn => {
                if c == Color::Red {
                    PST_PAWN_RED[y_usize][x_usize]
                } else {
                    PST_PAWN_RED[y_mirrored][x_usize]
                }
            }
            PieceType::Horse => {
                if c == Color::Red {
                    PST_HORSE_RED[y_usize][x_usize]
                } else {
                    PST_HORSE_RED[y_mirrored][x_usize]
                }
            }
            PieceType::Chariot => {
                if c == Color::Red {
                    PST_CHARIOT_RED[y_usize][x_usize]
                } else {
                    PST_CHARIOT_RED[y_mirrored][x_usize]
                }
            }
            _ => 0,
        };

        if c == Color::Red { val } else { -val }
    }

    pub fn evaluate(board: &Board, side: Color) -> i32 {
        let mut score = 0;

        for y in 0..10 {
            for x in 0..9 {
                if let Some(piece) = board.cells[y][x] {
                    let base = base_value(piece.piece_type);
                    let pos = position_value(piece.piece_type, piece.color, x as u8, y as u8);
                    let total = base + pos;

                    if piece.color == Color::Red {
                        score += total;
                    } else {
                        score -= total;
                    }
                }
            }
        }

        if side == Color::Red { score } else { -score }
    }
}

// 走法生成模块
mod movegen {
    use super::*;

    fn generate_pawn_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dir = if color == Color::Red { -1 } else { 1 };

        // 前进
        let forward = Coord::new(pos.x, (pos.y as i8 + dir) as u8);
        if forward.is_valid() {
            if let Some(p) = board.get(forward) {
                if p.color != color {
                    moves.push(forward);
                }
            } else {
                moves.push(forward);
            }
        }

        // 过河后横走
        if pos.crosses_river(color) {
            for dx in [-1, 1].iter() {
                let side = Coord::new((pos.x as i8 + dx) as u8, pos.y);
                if side.is_valid() {
                    if let Some(p) = board.get(side) {
                        if p.color != color {
                            moves.push(side);
                        }
                    } else {
                        moves.push(side);
                    }
                }
            }
        }

        moves
    }

    fn generate_horse_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [
            (2, 1, 1, 0),
            (2, -1, 1, 0),
            (-2, 1, -1, 0),
            (-2, -1, -1, 0),
            (1, 2, 0, 1),
            (1, -2, 0, -1),
            (-1, 2, 0, 1),
            (-1, -2, 0, -1),
        ];

        for (dx, dy, bx, by) in deltas.iter() {
            let tar = Coord::new((pos.x as i8 + dx) as u8, (pos.y as i8 + dy) as u8);
            let block = Coord::new((pos.x as i8 + bx) as u8, (pos.y as i8 + by) as u8);

            if tar.is_valid() && board.get(block).is_none() {
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        moves.push(tar);
                    }
                } else {
                    moves.push(tar);
                }
            }
        }

        moves
    }

    fn generate_chariot_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        for (dx, dy) in dirs.iter() {
            let mut x = pos.x as i8 + dx;
            let mut y = pos.y as i8 + dy;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x as u8, y as u8);
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        moves.push(tar);
                    }
                    break;
                } else {
                    moves.push(tar);
                }
                x += dx;
                y += dy;
            }
        }

        moves
    }

    fn generate_cannon_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        for (dx, dy) in dirs.iter() {
            let mut x = pos.x as i8 + dx;
            let mut y = pos.y as i8 + dy;
            let mut jumped = false;

            while x >= 0 && x < 9 && y >= 0 && y < 10 {
                let tar = Coord::new(x as u8, y as u8);
                if let Some(p) = board.get(tar) {
                    if jumped {
                        if p.color != color {
                            moves.push(tar);
                        }
                        break;
                    } else {
                        jumped = true;
                    }
                } else {
                    if !jumped {
                        moves.push(tar);
                    }
                }
                x += dx;
                y += dy;
            }
        }

        moves
    }

    fn generate_elephant_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [
            (2, 2, 1, 1),
            (2, -2, 1, -1),
            (-2, 2, -1, 1),
            (-2, -2, -1, -1),
        ];

        for (dx, dy, bx, by) in deltas.iter() {
            let tar = Coord::new((pos.x as i8 + dx) as u8, (pos.y as i8 + dy) as u8);
            let block = Coord::new((pos.x as i8 + bx) as u8, (pos.y as i8 + by) as u8);

            if tar.is_valid() && !tar.crosses_river(color) && board.get(block).is_none() {
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        moves.push(tar);
                    }
                } else {
                    moves.push(tar);
                }
            }
        }

        moves
    }

    fn generate_advisor_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (dx, dy) in deltas.iter() {
            let tar = Coord::new((pos.x as i8 + dx) as u8, (pos.y as i8 + dy) as u8);
            if tar.is_valid() && tar.in_palace(color) {
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        moves.push(tar);
                    }
                } else {
                    moves.push(tar);
                }
            }
        }

        moves
    }

    fn generate_king_moves(board: &Board, pos: Coord, color: Color) -> Vec<Coord> {
        let mut moves = Vec::new();
        let deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        for (dx, dy) in deltas.iter() {
            let tar = Coord::new((pos.x as i8 + dx) as u8, (pos.y as i8 + dy) as u8);
            if tar.is_valid() && tar.in_palace(color) {
                if let Some(p) = board.get(tar) {
                    if p.color != color {
                        moves.push(tar);
                    }
                } else {
                    moves.push(tar);
                }
            }
        }

        moves
    }

    pub fn generate_all_moves(board: &Board, color: Color) -> Vec<Action> {
        let mut actions = Vec::new();

        for y in 0..10 {
            for x in 0..9 {
                let pos = Coord::new(x as u8, y as u8);
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
                            let captured = board.get(tar);
                            actions.push(Action::new(pos, tar, captured));
                        }
                    }
                }
            }
        }

        actions
    }
}

// 搜索模块
mod search {
    use super::*;
    use eval::evaluate;
    use movegen::generate_all_moves;

    pub fn alpha_beta(
        board: &mut Board,
        initial_depth: u8,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        side: Color,
        best_action: &mut Option<Action>,
    ) -> i32 {
        assert!(initial_depth >= depth);
        if depth == 0 {
            return evaluate(board, side);
        }

        let mut moves = generate_all_moves(board, side);

        // 简单的启发式排序：吃子优先
        moves.sort_by_key(|a| if a.captured.is_some() { 0 } else { 1 });

        if moves.is_empty() {
            return -100000 + ((initial_depth - depth) as i32); // 输了，但晚输比早输好
        }

        let mut max_eval = -i32::MAX;

        for action in moves {
            board.make_move(action);

            let illegal = board.is_face_to_face();

            let eval = if illegal {
                -100000
            } else {
                -alpha_beta(
                    board,
                    initial_depth,
                    depth - 1,
                    -beta,
                    -alpha,
                    side.opponent(),
                    &mut None,
                )
            };

            board.undo_move(action);

            if illegal {
                continue;
            }

            if eval > max_eval {
                max_eval = eval;
                if best_action.is_some() {
                    *best_action = Some(action);
                }
            }

            if eval > alpha {
                alpha = eval;
            }

            if alpha >= beta {
                break;
            }
        }

        max_eval
    }

    pub fn find_best_move(board: &mut Board, depth: u8, side: Color) -> Option<Action> {
        let mut best_action = Some(Action::new(Coord::new(0, 0), Coord::new(0, 0), None));
        let _ = alpha_beta(board, depth, depth, -100000, 100000, side, &mut best_action);
        best_action
    }
}

pub const MAX_DEPTH: u8 = 6;
macro_rules! next_move {
    ($board:ident) => {
        // AI走 (红方)
        println!("AI思考中...");
        if let Some(action) = search::find_best_move(&mut $board, MAX_DEPTH, Color::Red) {
            println!(
                "AI走: ({}, {}) -> ({}, {})",
                action.src.x, action.src.y, action.tar.x, action.tar.y
            );
            $board.make_move(action);
            println!("{}", $board);
        } else {
            println!("AI无子可走，你赢了！");
            break;
        }
    };
}

macro_rules! reinput {
    ($ai_move:ident) => {
        $ai_move = false;
        continue;
    };
}
fn main() {
    use std::io;

    let mut board = Board::new();
    println!("中国象棋引擎");
    println!("{}", board);

    let stdin = io::stdin();
    let mut input = String::new();

    println!("请选择落子顺序: 1.AI先 2.玩家先");
    stdin.read_line(&mut input).unwrap();
    let order = input.trim().parse::<u8>().unwrap();
    if !(1..=2).contains(&order) {
        panic!("Invalid order number");
    }
    let mut ai_move = true;

    loop {
        if order == 1 {
            if ai_move {
                next_move!(board);
            }
        }

        // 玩家走 (黑方)
        println!("请输入你的走法 (格式: x1 y1 x2 y2):");
        input.clear();
        stdin.read_line(&mut input).unwrap();
        let parts: Vec<&str> = input.trim().split_whitespace().collect();

        if parts.len() != 4 {
            println!("输入格式错误，请重新输入");
            reinput!(ai_move);
        }

        let x1 = match parts[0].parse::<u8>() {
            Ok(v) => v,
            Err(_) => {
                println!("无效坐标");
                reinput!(ai_move);
            }
        };
        let y1 = match parts[1].parse::<u8>() {
            Ok(v) => v,
            Err(_) => {
                println!("无效坐标");
                reinput!(ai_move);
            }
        };
        let x2 = match parts[2].parse::<u8>() {
            Ok(v) => v,
            Err(_) => {
                println!("无效坐标");
                reinput!(ai_move);
            }
        };
        let y2 = match parts[3].parse::<u8>() {
            Ok(v) => v,
            Err(_) => {
                println!("无效坐标");
                reinput!(ai_move);
            }
        };

        let src = Coord::new(x1, y1);
        let tar = Coord::new(x2, y2);

        if !src.is_valid() || !tar.is_valid() {
            println!("坐标超出棋盘");
            reinput!(ai_move);
        }

        match board.get(src) {
            Some(p) if p.color == Color::Black => {}
            _ => {
                println!("该位置没有你的棋子");
                reinput!(ai_move);
            }
        };

        let captured = board.get(tar);
        let action = Action::new(src, tar, captured);

        // 验证走法合法性
        let valid_moves = movegen::generate_all_moves(&board, Color::Black);
        if !valid_moves.iter().any(|&m| m.src == src && m.tar == tar) {
            println!("非法走法");
            reinput!(ai_move);
        }

        board.make_move(action);

        if board.is_face_to_face() {
            println!("将帅不能照面，非法走法");
            board.undo_move(action);
            reinput!(ai_move);
        }

        println!("{}", board);
        ai_move = true;

        if order == 2 {
            if ai_move {
                next_move!(board);
            }
        }
    }
}
