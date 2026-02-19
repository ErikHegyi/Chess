use std::{
    path::Path,
    fs::write,
    io::Result
};
use crate::{
    board::ChessBoard,
    board_mask::BoardMask,
    player::Player
};
use serde_json;
use serde::{Deserialize, Serialize};
use tch::Tensor;


#[derive(Serialize, Deserialize, Clone)]
pub struct Game {
    dimensions: Vec<usize>,
    mask: BoardMask,
    past_positions: Vec<ChessBoard>
}

impl Game {
    pub fn new(dimensions: Vec<usize>) -> Self {
        // Only allow 2D and 3D chess
        let spatial_dimensions: usize = dimensions.len();
        assert!(spatial_dimensions == 2 || spatial_dimensions == 3);

        // Create a mask making each tile legal
        let mask: BoardMask = match spatial_dimensions {
            2 => BoardMask::BoardMask2D(vec![vec![true; dimensions[1]]; dimensions[0]]),
            3 => BoardMask::BoardMask3D(vec![vec![vec![true; dimensions[2]]; dimensions[1]]; dimensions[0]]),
            _ => panic!("This should have been impossible")
        };
        
        // Create the chessboard
        let chessboard: ChessBoard = ChessBoard::new();
        
        // Create a vector containing the past positions
        let past_positions: Vec<ChessBoard> = vec![chessboard];

        Self {
            dimensions,
            mask,
            past_positions
        }
    }
    
    pub fn dim(&self) -> &Vec<usize> { &self.dimensions }
    pub fn current_position(&self) -> &ChessBoard {
        match self.past_positions.last() {
            Some(board) => board,
            None => panic!("Please don't hack the system")
        }
    }
    pub fn first_position(&self) -> &ChessBoard {
        match self.past_positions.first() {
            Some(board) => board,
            None => panic!("No way this keeps happening")
        }
    }

    pub fn apply_mask(&mut self, mask: BoardMask) { self.mask = mask; }

    pub fn export(&self, path: &Path) -> Result<()> {
        // Convert to JSON
        let json: String = serde_json::to_string_pretty(&self)?;

        // Export
        write(path, json.as_bytes())
    }
    
    pub fn state_tensor(&self, move_number: usize) -> Tensor { todo!() }
    pub fn is_over(&self) -> bool { todo!() }
    pub fn get_players(&self) -> &Vec<Player> { self.first_position().players() }
    pub fn number_of_players(&self) -> usize { self.get_players().len() }
    pub fn possible_moves_for_player(&self, player: &Player) -> Vec<ChessBoard> { todo!() }
    pub fn new_move(&mut self, new_move: ChessBoard) { self.past_positions.push(new_move) }
    pub fn unmake_move(&mut self) { self.past_positions.pop().unwrap(); }
    pub fn winner(&mut self) -> usize { todo!() }
}