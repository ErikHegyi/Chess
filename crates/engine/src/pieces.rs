use crate::board::ChessBoard;
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Clone)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King
}

// Shorthand types
type Position = Vec<usize>;
type MoveVector = Vec<Position>;


#[derive(Serialize, Deserialize, Clone)]
pub struct Piece {
    id: usize,
    piece_type: PieceType,
    moves: usize,
    position: Position
}

impl Piece {
    pub fn new(
        id: usize,
        piece_type: PieceType,
        position: Position
    ) -> Self {
        Self {
            id,
            piece_type,
            moves: 0,
            position
        }
    }

    pub fn id(&self) -> usize { self.id }
    pub fn pos(&self) -> &Vec<usize> { &self.position }

    pub fn move_to(&mut self, position: Position) {
        // The dimensionality must be the same
        assert_eq!(self.position.len(), position.len());

        // Move the piece
        self.position = position;
    }

    pub fn legal_moves(&self, board: &ChessBoard) -> MoveVector {
        match self.piece_type {
            PieceType::Pawn => self.legal_moves_pawn(board),
            _ => todo!()
        }
    }

    fn legal_moves_pawn(&self, board: &ChessBoard) -> MoveVector {
        todo!()
    }
}