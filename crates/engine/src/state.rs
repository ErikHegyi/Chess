use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tch::Tensor;

pub type Position = Vec<usize>;
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum PieceType { Pawn, Knight, Bishop, Rook, Queen, King }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Piece { pub piece_type: PieceType, pub player: usize, pub first_move: bool }
impl Piece {
    pub fn new(piece_type: PieceType, player: usize) -> Self {
        Self { piece_type, player, first_move: true }
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Move { pub player: usize, pub from: Position, pub to: Position,
    pub promotion: Option<PieceType> }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct State {
    pub starting_board: Vec<Option<Piece>>, pub board: Vec<Option<Piece>>, pub mask: Vec<bool>,
    pub player_to_move: usize, pub number_of_players: usize, pub moves: Vec<Move>,
    pub last_capture: usize,
}
pub trait StateTensor {
    fn state_tensor(&self) -> Tensor where Self: Variant;
    fn state_tensor_with_move(&mut self, movement: Move) -> Tensor where Self: Variant;
    fn dimensions() -> Vec<usize> where Self: Variant;
}
pub trait Variant {
    fn new() -> Self;
    fn folder_name() -> String;
    fn possible_moves(&self, player: usize) -> Vec<Move>;
    fn legal_moves(&self, player: usize) -> Vec<Move>;
    fn is_in_check(&self, player: usize, position: Position) -> bool;
    fn is_king_in_check(&self, player: usize) -> bool;
    fn move_piece(&mut self, movement: Move);
    fn number_of_pieces() -> usize;
    fn number_of_players() -> usize;
    fn player_to_move(&self) -> usize;
    fn number_of_moves(&self) -> usize;
    fn past_moves(&self) -> &Vec<Move>;
    fn winner(&mut self) -> Option<i64>;
}


pub trait Export {
    fn export(&self, path: &Path) -> std::io::Result<()> where Self: Serialize {
        // Convert to JSON
        let json: String = serde_json::to_string_pretty(&self)?;

        // Export
        fs::write(path, json.as_bytes())
    }
}