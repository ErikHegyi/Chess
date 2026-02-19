use serde::{Deserialize, Serialize};
use crate::pieces::Piece;


#[derive(Serialize, Deserialize, Clone)]
pub struct Player {
    pieces: Vec<Piece>,
    is_in_the_game: bool
}


impl Player {
    pub fn new() -> Self { Self { pieces: Vec::new(), is_in_the_game: true } }
    pub fn add_piece(&mut self, piece: Piece) { self.pieces.push(piece) }
    pub fn is_in_the_game(&self) -> bool { self.is_in_the_game }
}