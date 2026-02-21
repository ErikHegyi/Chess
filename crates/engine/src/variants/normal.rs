use serde::{Deserialize, Serialize};
use tch::{IndexOp, Tensor};
use crate::state::*;


const WIDTH: usize = 8;
const HEIGHT: usize = 8;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NormalChess {
    variant: String,
    dimensions: Vec<usize>,
    state: State
}


impl StateTensor for NormalChess {
    fn state_tensor(&self) -> Tensor {
        // Get the dimensions
        let channel_dimensions: usize = Self::number_of_players() * Self::number_of_pieces() + 1;

        // Get the dimensions
        let dimensions: Vec<i64> = vec![channel_dimensions as i64, WIDTH as i64, HEIGHT as i64];

        // Get the device
        let device: tch::Device = tch::Device::cuda_if_available();

        // Create the tensor
        let mut tensor: Tensor = Tensor::zeros(&dimensions, (tch::Kind::Double, device));

        // Create a vector of piece types so we can look up the dimensions
        let piece_type_vector: Vec<PieceType> = vec![
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King
        ];

        // Safety check (in case of some changes in the future
        assert_eq!(piece_type_vector.len(), Self::number_of_pieces());

        // Iterate through each player
        for i in 0..self.state.number_of_players {
            // Iterate through each piece
            for (j, tile) in self.state.board.iter().enumerate() {
                // Unwrap
                let piece: &Piece = match tile {
                    Some(p) => p, None => continue
                };

                // Calculate the dimension
                let pos: usize = match piece_type_vector.iter().position(|x| *x == piece.piece_type) {
                    Some(u) => u,
                    None => panic!("The piece was not found in the piece type vector.")
                };
                let dim: usize = i * piece_type_vector.len() + pos;

                // Update the dimensions
                tensor
                    .i((dim as i64, (j / WIDTH) as i64, (j % WIDTH) as i64))
                    .fill_(1.0);
            }
        }

        // Apply the mask
        let mask_tensor: Tensor = {
            let flat: Vec<i64> = vec![1; WIDTH * HEIGHT];
            Tensor::from_slice(&flat).view([HEIGHT as i64, WIDTH as i64])
        };
        tensor.narrow(0, channel_dimensions as i64 - 1, 1).copy_(&mask_tensor);
        tensor
    }
    fn state_tensor_with_move(&mut self, movement: Move) -> Tensor {
        // Calculate the indices
        let from: usize = movement.from[0] * WIDTH + movement.from[1];
        let to: usize = movement.to[0] * WIDTH + movement.to[1];

        // Check if any piece has been taken
        let piece_taken: Option<Piece> = self.state.board[to].clone();

        // Make the step
        self.move_piece(movement.clone());

        // Get the state tensor
        let state_tensor: Tensor = self.state_tensor();

        // Un-make the move
        self.state.moves.pop().unwrap();  // Remove from the vector
        self.state.board[from] = self.state.board[to].clone();  // Put the moved piece back;
        self.state.board[to] = piece_taken;  // If a has been taken, put it back

        state_tensor
    }
    fn dimensions() -> Vec<usize> { vec![Self::number_of_players() * Self::number_of_pieces() + 1, HEIGHT, WIDTH] }
}

impl Variant for NormalChess {
    fn new() -> Self {
        // Create the board vector
        let mut board: Vec<Option<Piece>> = vec![None; WIDTH * HEIGHT];

        // Create the pieces for each player
        for i in 0usize..2usize {
            // Create 8 pawns
            let pawns: Vec<Piece> = vec![Piece::new(PieceType::Pawn, i); 8];

            // Create two rooks
            let rooks: Vec<Piece> = vec![Piece::new(PieceType::Rook, i); 2];

            // Create two knights
            let knights: Vec<Piece> = vec![Piece::new(PieceType::Knight, i); 2];

            // Create two bishops
            let bishops: Vec<Piece> = vec![Piece::new(PieceType::Bishop, i); 2];

            // Create a queen
            let queen: Piece = Piece::new(PieceType::Queen, i);

            // Create a king
            let king: Piece = Piece::new(PieceType::King, i);

            // Get the rows
            let rows: (usize, usize) = match i {
                0 => (1, 0),
                1 => (HEIGHT - 2, HEIGHT - 1),
                _ => panic!("This should not have been possible")
            };

            // Add the pawns
            let starting_index: usize = rows.0 * WIDTH;
            let mut j: usize = 0;
            for pawn in pawns {
                board[starting_index + j] = Some(pawn);
                j += 1;
            }

            // Add the rooks
            let starting_index: usize = rows.1 * WIDTH;
            let mut j: usize = 0;
            for rook in rooks {
                board[starting_index + j * (WIDTH - 1)] = Some(rook);
                j += 1;
            }

            // Add the knights
            let mut j: usize = 0;
            for knight in knights {
                board[starting_index + 1 + j * (WIDTH - 3)] = Some(knight);
                j += 1;
            }

            // Add the bishops
            let mut j: usize = 0;
            for bishop in bishops {
                board[starting_index + 2 + j * (WIDTH - 5)] = Some(bishop);
                j += 1;
            }

            // Add the queen
            board[starting_index + 3] = Some(queen);

            // Add the king
            board[starting_index + (WIDTH - 4)] = Some(king);
        }

        // Create the mask
        // In normal chess, all tiles are legal
        let mask: Vec<bool> = vec![true; WIDTH * HEIGHT];

        // Set the player to start
        let player_to_move: usize = 0;

        // Set the number of players
        let number_of_players: usize = 2;

        // Set the history of moves to empty
        let moves: Vec<Move> = Vec::new();

        // Set the move number of the last capture to 0
        let last_capture: usize = 0;

        // Create the state
        let state: State = State {
            starting_board: board.clone(),
            board,
            mask,
            player_to_move,
            number_of_players,
            moves,
            last_capture
        };

        // Return the game struct
        Self { variant: "classic".to_string(), dimensions: vec![WIDTH, HEIGHT], state }
    }
    #[inline]
    fn folder_name() -> String { "normal".to_string() }
    fn possible_moves(&self, player: usize) -> Vec<Move> {
        let mut possible_moves: Vec<Move> = Vec::new();

        // Iterate through each piece
        for (index, piece) in self.state.board.iter().enumerate() {
            // Unwrap the piece
            match piece {
                Some(piece) => {
                    if player != piece.player { continue; }

                    // Calculate the coordinates
                    let row: usize = index / WIDTH;
                    let column: usize = index % WIDTH;

                    // Shorthand
                    let board: &Vec<Option<Piece>> = &self.state.board;

                    // Calculate the possible moves for this piece
                    match piece.piece_type {
                        PieceType::Pawn => {
                            // Calculate the direction
                            let dir: i8 = if player == 0 { 1 } else { -1 };

                            // Can not go out of the board
                            if (row == 0 && dir == -1) || (row == HEIGHT - 1 && dir == 1) {
                                continue;
                            }

                            // Calculate the new coordinates
                            let new_row: usize = (row as i8 + dir) as usize;
                            let new_column: usize = column;

                            // Test if there is anything blocking its path
                            match board[new_row * WIDTH + new_column] {
                                Some(_) => continue,
                                None => {
                                    // Test whether it can promote
                                    if (player == 0 && new_row == HEIGHT - 1) || (player == 1 && new_row == 0) {
                                        for piece_type in [
                                            PieceType::Knight,
                                            PieceType::Bishop,
                                            PieceType::Rook,
                                            PieceType::Queen
                                        ] {
                                            possible_moves.push(
                                                Move {
                                                    player,
                                                    from: vec![row, column],
                                                    to: vec![new_row, new_column],
                                                    promotion: Some(piece_type)
                                                }
                                            )
                                        }
                                    } else {
                                        possible_moves.push(
                                            Move {
                                                player,
                                                from: vec![row, column],
                                                to: vec![new_row, new_column],
                                                promotion: None
                                            }
                                        )
                                    }

                                    // Test whether this is the first move
                                    if piece.first_move {
                                        if (row == 1 && dir == -1) || (row == HEIGHT - 2 && dir == 1) {
                                            continue;
                                        }

                                        // Calculate the new coordinates
                                        let new_row: usize = (row as i8 + 2 * dir) as usize;
                                        let new_column: usize = column;

                                        // Test whether there is anything blocking its path
                                        match board[new_row * WIDTH + new_column] {
                                            Some(_) => continue,
                                            None => possible_moves.push(
                                                Move {
                                                    player,
                                                    from: vec![row, column],
                                                    to: vec![new_row, new_column],
                                                    promotion: None
                                                }
                                            )
                                        }
                                    }
                                }
                            }

                            // Test whether it can take anything
                            for i in [-1, 1] {
                                let new_row: i8 = row as i8 + dir;
                                let new_column: i8 = column as i8 + i;

                                if new_column >= 0 && new_column < WIDTH as i8 {
                                    // Calculate the index
                                    let index: usize = (new_row as usize) * WIDTH + (new_column as usize);

                                    // Check whether any other piece is there
                                    match &board[index] {
                                        Some(other) => {
                                            // Can only take if the piece does not belong to the
                                            // currently selected player
                                            if other.player != player {
                                                // Test whether it can promote
                                                if (player == 0 && new_row as usize == HEIGHT - 1) || (player == 1 && new_row == 0) {
                                                    for piece_type in [
                                                        PieceType::Knight,
                                                        PieceType::Bishop,
                                                        PieceType::Rook,
                                                        PieceType::Queen
                                                    ] {
                                                        possible_moves.push(
                                                            Move {
                                                                player,
                                                                from: vec![row, column],
                                                                to: vec![new_row as usize, new_column as usize],
                                                                promotion: Some(piece_type)
                                                            }
                                                        )
                                                    }
                                                } else {
                                                    possible_moves.push(
                                                        Move {
                                                            player,
                                                            from: vec![row, column],
                                                            to: vec![new_row as usize, new_column as usize],
                                                            promotion: None
                                                        }
                                                    )
                                                }
                                            }
                                        },
                                        None => continue
                                    }
                                }
                            }

                            // En passant
                            for i in [-1, 1] {
                                let new_row: i8 = row as i8 + dir;
                                let new_column: i8 = column as i8 + i;

                                if new_column >= 0 && new_column < WIDTH as i8 {
                                    // Calculate the index
                                    let index: usize = ((new_row - dir) as usize) * WIDTH + (new_column as usize);

                                    // Check whether any other piece is there
                                    match &board[index] {
                                        Some(other) => {
                                            // Can only take if the piece does not belong to the
                                            // currently selected player
                                            if other.player != player {
                                                // Can only take a pawn
                                                if other.piece_type == PieceType::Pawn {
                                                    // Get the moves in the last round
                                                    let moves_to_check: &[Move] = &self.state.moves[
                                                        self.state.moves.len() - self.state.number_of_players + 1..
                                                    ];

                                                    // Check each move
                                                    for m in moves_to_check {
                                                        if m.to == vec![row, new_column as usize] {
                                                            if m.from == vec![(row as i8 + 2 * dir) as usize, new_column as usize] {
                                                                possible_moves.push(
                                                                    Move {
                                                                        player,
                                                                        from: vec![row, column],
                                                                        to: vec![new_row as usize, new_column as usize],
                                                                        promotion: None
                                                                    }
                                                                )
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        None => continue
                                    }
                                }
                            }
                        },
                        PieceType::Knight => {
                            // Define the offsets
                            let offsets: [(i8, i8); 8] = [
                                (-2, 1),
                                (-1, 2),
                                (1, 2),
                                (2, 1),
                                (2, -1),
                                (1, -2),
                                (-1, -2),
                                (-2, -1)
                            ];

                            // Check each offset
                            for (x, y) in offsets {
                                let new_row: i8 = row as i8 + y;
                                let new_column: i8 = column as i8 + x;
                                if 0 <= new_column && new_column < WIDTH as i8 {
                                    if 0 <= new_row && new_row < HEIGHT as i8 {
                                        match &board[(new_row as usize) * WIDTH + (new_column as usize)] {
                                            Some(other) => {
                                                if other.player != player {
                                                    possible_moves.push(
                                                        Move {
                                                            player,
                                                            from: vec![row, column],
                                                            to: vec![new_row as usize, new_column as usize],
                                                            promotion: None
                                                        }
                                                    )
                                                }
                                            },
                                            None => {
                                                possible_moves.push(
                                                    Move {
                                                        player,
                                                        from: vec![row, column],
                                                        to: vec![new_row as usize, new_column as usize],
                                                        promotion: None
                                                    }
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        PieceType::King => {
                            // Define the offsets
                            let offsets: [(i8, i8); 8] = [
                                (-1, -1),
                                (-1, 0),
                                (-1, 1),
                                (0, -1),
                                (0, 1),
                                (1, -1),
                                (1, 0),
                                (1, 1)
                            ];

                            // Check each offset
                            for (x, y) in offsets {
                                let new_row: i8 = row as i8 + y;
                                let new_column: i8 = column as i8 + x;
                                if 0 <= new_column && new_column < WIDTH as i8 {
                                    if 0 <= new_row && new_row < HEIGHT as i8 {
                                        match &board[(new_row as usize) * WIDTH + (new_column as usize)] {
                                            Some(other) => {
                                                if other.player != player {
                                                    possible_moves.push(
                                                        Move {
                                                            player,
                                                            from: vec![row, column],
                                                            to: vec![new_row as usize, new_column as usize],
                                                            promotion: None
                                                        }
                                                    )
                                                }
                                            },
                                            None => {
                                                possible_moves.push(
                                                    Move {
                                                        player,
                                                        from: vec![row, column],
                                                        to: vec![new_row as usize, new_column as usize],
                                                        promotion: None
                                                    }
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        _ => {
                            let directions: Vec<(i8, i8)> = match piece.piece_type {
                                PieceType::Bishop => vec![(1,1), (1,-1), (-1,1), (-1,-1)],
                                PieceType::Rook   => vec![(0,1), (0,-1), (1,0), (-1,0)],
                                PieceType::Queen  => vec![(1,1), (1,-1), (-1,1), (-1,-1), (0,1), (0,-1), (1,0), (-1,0)],
                                _ => vec![]
                            };

                            for (dr, dc) in directions {
                                let mut i = 1;
                                loop {
                                    let new_r = row as i8 + (dr * i);
                                    let new_c = column as i8 + (dc * i);

                                    if new_r < 0 || new_r >= HEIGHT as i8 || new_c < 0 || new_c >= WIDTH as i8 { break; }

                                    let target_idx = (new_r as usize) * WIDTH + (new_c as usize);
                                    match &board[target_idx] {
                                        Some(other) => {
                                            if other.player != player {
                                                possible_moves.push(Move { player, from: vec![row, column], to: vec![new_r as usize, new_c as usize], promotion: None });
                                            }
                                            break; // Hit a piece, stop sliding
                                        },
                                        None => {
                                            possible_moves.push(Move { player, from: vec![row, column], to: vec![new_r as usize, new_c as usize], promotion: None });
                                        }
                                    }
                                    i += 1;
                                }
                            }
                        }
                    }
                },
                None => continue
            }
        }

        possible_moves
    }
    fn legal_moves(&self, player: usize) -> Vec<Move> {
        // Filter out moves that threaten the king
        let possible_moves: Vec<Move> = self.possible_moves(player);
        let mut legal_moves: Vec<Move> = Vec::new();

        for m in possible_moves {
            let mut temp_game: Self = self.clone();
            temp_game.move_piece(m.clone());
            if !temp_game.is_king_in_check(player) {
                legal_moves.push(m);
            }
        }

        legal_moves
    }

    fn is_in_check(&self, player: usize, position: Position) -> bool {
        for other in 0..self.state.number_of_players {
            // Ignore checks coming from the same player
            if player == other { continue; }

            // Get each possible move
            for possible_move in self.possible_moves(other) {
                if possible_move.to == position { return true; }
            }
        }
        false
    }
    fn is_king_in_check(&self, player: usize) -> bool {
        // Look for the player's king
        for (i, tile) in self.state.board.iter().enumerate() {
            match tile {
                Some(piece) => {
                    if piece.piece_type == PieceType::King && piece.player == player {
                        if self.is_in_check(player, vec![i / WIDTH, i % WIDTH]) {
                            return true;
                        }
                    }
                },
                None => continue
            }
        }
        false
    }

    fn move_piece(&mut self, movement: Move) {
        // Calculate the indices
        let from: usize = movement.from[0] * WIDTH + movement.from[1];
        let to: usize = movement.to[0] * WIDTH + movement.to[1];

        // Clone the piece
        let mut piece: Piece = match &self.state.board[from] {
            Some(piece) => piece.clone(),
            None => panic!("I have no idea how this could have ever happened")
        };

        // Set the first move variable to false, since it has now moved
        piece.first_move = false;

        // Set which side is to move
        self.state.player_to_move = 1 - piece.player;

        // Remove the piece from its original position
        self.state.board[from] = None;

        // En passant
        if piece.piece_type == PieceType::Pawn {
            // Changed columns
            if movement.from[1] != movement.to[1] && self.state.board[to].is_none() {
                // Get the direction
                let dir: i32 = (movement.to[0] as i32 - movement.from[0] as i32) / (movement.to[0] as i32 - movement.from[0] as i32).abs();

                // Check whether there is a pawn next to it
                if !((movement.to[0] as i32 - dir) < 0 || (movement.to[0] as i32 - dir) >= HEIGHT as i32) {
                    let index_to_check: i32 = (movement.to[0] as i32 - dir) * WIDTH as i32 + movement.to[1] as i32;

                    match &self.state.board[index_to_check as usize] {
                        Some(other) => {
                            if other.piece_type == PieceType::Pawn {
                                // Remove
                                self.state.board[index_to_check as usize] = None;
                            }
                        },
                        None => ()
                    }
                }
            }
        }

        // Make the piece appear somewhere else
        self.state.last_capture = if self.state.board[to].is_some() { 0 } else { self.state.last_capture + 1 };
        self.state.board[to] = if movement.promotion.is_none() { Some(piece) } else {
            piece.piece_type = movement.promotion.unwrap();
            Some(piece)
        };

        // Append the move
        self.state.moves.push(movement);
    }
    #[inline]
    fn number_of_pieces() -> usize { 6 }
    #[inline]
    fn number_of_players() -> usize { 2 }
    #[inline]
    fn player_to_move(&self) -> usize { self.state.player_to_move }
    #[inline]
    fn number_of_moves(&self) -> usize { self.state.moves.len() }
    #[inline]
    fn past_moves(&self) -> &Vec<Move> { &self.state.moves }
    fn winner(&mut self) -> Option<i64> {
        // Get the player
        let player: usize = self.state.player_to_move;

        // Check whether the player has been checkmated
        if self.is_king_in_check(player) && self.legal_moves(player).len() == 0 {
            match player {
                0 => Some(1),
                1 => Some(0),
                _ => panic!("A player has entered the arena.")
            }
        }

        // No legal moves, but the king is not in check => DRAW
        else if self.legal_moves(player).len() == 0 {
            Some(-1)
        }

        // No captures have happened for the past 50 moves
        else if self.state.moves.len() >= 100 && self.state.last_capture <= self.state.moves.len() - 100 {
            Some(-1)
        }

        else {
            // Repetition
            let moves: &Vec<Move> = &self.state.moves;  // Shorthand notation
            let l: usize = moves.len();
            if l >= 8
                && moves[l - 1] == moves[l - 5]
                && moves[l - 2] == moves[l - 6]
                && moves[l - 3] == moves[l - 7]
                && moves[l - 4] == moves[l - 8] {
                Some(-1)
            }

            // The game has not yet ended
            else {
                None
            }
        }
    }
}


impl Export for NormalChess {}