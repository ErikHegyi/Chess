use std::{
    path::Path,
    fs::write,
    io::Result
};
use crate::{board::ChessBoard, board_mask::BoardMask, player::Player, PieceType};
use serde_json;
use serde::{Deserialize, Serialize};
use tch::{IndexOp, Tensor};


#[derive(Serialize, Deserialize, Clone)]
pub struct Game {
    dimensions: Vec<usize>,
    mask: BoardMask,
    past_positions: Vec<ChessBoard>,
    pieces: Vec<PieceType>
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
            past_positions,
            pieces: Vec::new()
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

    pub fn state_tensor(&self, move_number: usize) -> Tensor {
        // Get the dimensions
        let channel_dimensions: usize = self.number_of_players() * self.get_players()[0].number_of_pieces() + 1;

        // Get the dimensions
        let mut dimensions: Vec<i64> = self.dim().iter().map(|x| *x as i64).collect();
        dimensions.insert(0, channel_dimensions as i64);

        // Get the device
        let device: tch::Device = tch::Device::cuda_if_available();

        // Create the tensor
        let mut tensor: Tensor = Tensor::zeros(&dimensions, (tch::Kind::Double, device));

        // Iterate through each player
        let mut n: i64 = 0;  // Keep track of the dimensions
        for (i, player) in self.past_positions[move_number].players().iter().enumerate() {
            // Iterate through each piece the player has
            for (j, piece_type) in self.pieces.iter().enumerate() {
                // Get the dimensions
                let dim_index: usize = i * self.pieces.len() + j;

                n += 1;

                // Iterate through the player's pieces
                for piece in player.get_pieces() {
                    // Check if the piece is of the type that we are looking for
                    if piece.piece_type() == *piece_type {
                        // Get the piece's position
                        let piece_pos: &Vec<usize> = piece.pos();

                        // Update the tensor
                        match piece_pos.len() {
                            2 => {
                                tensor = tensor
                                    .i((dim_index as i64, piece_pos[0] as i64, piece_pos[1] as i64))
                                    .fill_(1.0);
                            },
                            3 => {
                                tensor = tensor
                                    .i((dim_index as i64, piece_pos[0] as i64, piece_pos[1] as i64, piece_pos[2] as i64))
                                    .fill_(1.0);
                            },
                            _ => panic!("I report, Sir, we have hyperjumped into a new dimension.")
                        }
                    }
                }
            }
        }

        // Apply the mask
        let mask_tensor: Tensor = match &self.mask {
            BoardMask::BoardMask2D(v) => {
                // Flatten Vec<Vec<bool>> → Vec<i64>
                let flat: Vec<i64> = v.iter()
                    .flat_map(|row| row.iter().map(|&b| if b { 1 } else { 0 }))
                    .collect();

                // Shape as [rows, cols]
                let rows = v.len() as i64;
                let cols = v[0].len() as i64;

                Tensor::from_slice(&flat).view([rows, cols])
            },
            BoardMask::BoardMask3D(v) => {
                // Flatten Vec<Vec<Vec<bool>>> → Vec<i64>
                let flat: Vec<i64> = v.iter()
                    .flat_map(|plane| plane.iter()
                        .flat_map(|row| row.iter().map(|&b| if b {1} else {0}))
                    )
                    .collect();

                let d1 = v.len() as i64;
                let d2 = v[0].len() as i64;
                let d3 = v[0][0].len() as i64;

                Tensor::from_slice(&flat).view([d1, d2, d3])
            }
        };
        tensor.narrow(1, n, 1).copy_(&mask_tensor);

        tensor
    }
    pub fn is_over(&self) -> bool { todo!() }
    pub fn get_players(&self) -> &Vec<Player> { self.first_position().players() }
    pub fn number_of_players(&self) -> usize { self.get_players().len() }
    pub fn possible_moves_for_player(&self, player: &Player) -> Vec<ChessBoard> { todo!() }
    pub fn new_move(&mut self, new_move: ChessBoard) { self.past_positions.push(new_move) }
    pub fn unmake_move(&mut self) { self.past_positions.pop().unwrap(); }
    pub fn winner(&mut self) -> usize { todo!() }
    pub fn set_board(&mut self, chess_board: ChessBoard) {
        // Only allow board setting if no moves have been made
        match self.past_positions.len() {
            1 => (),
            _ => panic!("Can only set the board before the first move")
        }

        // Set the board
        self.past_positions[0] = chess_board.clone();

        // Iterate through the players and their pieces
        // Save each new piece into a vector
        let mut piece_types: Vec<PieceType> = Vec::new();
        for player in chess_board.players() {
            for piece in player.get_pieces() {
                if !piece_types.contains(&piece.piece_type()) {
                    piece_types.push(piece.piece_type());
                }
            }
        }

        // Sort the pieces, so a dimensions always represents the same piece
        piece_types.sort();

        // Update the pieces
        self.pieces = piece_types;
    }
}