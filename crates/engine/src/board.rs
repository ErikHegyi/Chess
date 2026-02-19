use serde::{Deserialize, Serialize};
use crate::player::Player;

#[derive(Serialize, Deserialize, Clone)]
pub struct ChessBoard {
    players: Vec<Player>
}


impl ChessBoard {
    pub fn new() -> ChessBoard {
        // Create a vector for the players
        let players: Vec<Player> = Vec::new();

        // Return the chessboard
        Self {
            players
        }
    }

    pub fn players(&self) -> &Vec<Player> { &self.players }
    pub fn add_player(&mut self, player: Player) { self.players.push(player) }
}