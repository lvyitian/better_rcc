// Neural network evaluation stub - currently returns handcrafted evaluation
// Will be replaced with actual NN inference in future

use crate::{Board, Color};
use crate::eval::eval::handcrafted_evaluate;

pub fn nn_evaluate_or_handcrafted(board: &Board, side: Color, initiative: bool) -> i32 {
    // For now, just call the handcrafted evaluation
    // TODO: Replace with NN inference when model is loaded
    handcrafted_evaluate(board, side, initiative)
}
