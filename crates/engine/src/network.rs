use std::{
    fmt::Debug,
    iter::zip
};
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use tch::{
    Tensor,
    Device,
    nn::{
        ModuleT,
        Adam,
        Conv2D,
        Conv3D,
        Linear,
        Optimizer,
        OptimizerConfig
    },
    nn
};
use crate::{
    board::ChessBoard,
    game::Game,
    player::Player,
    hyperparameters::*
};


#[derive(Debug)]
struct Net2D {
    conv_1: Conv2D,
    conv_2: Conv2D,
    conv_3: Conv2D,
    conv_4: Conv2D,
    conv_5: Conv2D,
    conv_6: Conv2D,
    fc: Linear
}
impl Net2D {
    fn new(variable_store: &nn::Path, in_channels: i64) -> Self {
        // Configure the properties of the convolutional layers
        let conv_config: nn::ConvConfig = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };

        // Create the convolutional layers
        let conv_1: Conv2D = nn::conv2d(variable_store, in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_2: Conv2D = nn::conv2d(variable_store, in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_3: Conv2D = nn::conv2d(variable_store, in_channels, 32, KERNEL_SIZE, conv_config);
        let conv_4: Conv2D = nn::conv2d(variable_store, 32, 64, KERNEL_SIZE, conv_config);
        let conv_5: Conv2D = nn::conv2d(variable_store, 64, 128, KERNEL_SIZE, conv_config);
        let conv_6: Conv2D = nn::conv2d(variable_store, 128, 256, KERNEL_SIZE, conv_config);

        // Create the linear layer
        let fc: Linear = nn::linear(variable_store, 256, 1, Default::default());

        // Return the network
        Self { conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, fc}
    }
}


impl ModuleT for Net2D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual: &Tensor = xs;

        // Apply the first residual block
        let xs: &Tensor = &xs
            .apply(&self.conv_1)
            .relu()
            .apply(&self.conv_2);
        let xs: Tensor = (xs + residual).relu();

        // Apply another convolutional layer, this time without a residual block
        let xs: Tensor = xs
            .apply(&self.conv_3)
            .relu()
            .apply(&self.conv_4)
            .relu()
            .apply(&self.conv_5)
            .relu()
            .apply(&self.conv_6);

        // Apply GAP
        let xs: Tensor = xs
            .adaptive_avg_pool2d(&[1, 1])
            .view([xs.size()[0], -1]);

        // Convert to scalar
        let xs: Tensor = xs.apply(&self.fc);

        // Apply the final activation function
        xs.tanh()
    }
}


#[derive(Debug)]
struct Net3D {
    conv_1: Conv3D,
    conv_2: Conv3D,
    conv_3: Conv3D,
    conv_4: Conv3D,
    conv_5: Conv3D,
    conv_6: Conv3D,
    fc: Linear
}
impl Net3D {
    fn new(variable_store: &nn::Path, in_channels: i64) -> Self {
        // Configure the properties of the convolutional layers
        let conv_config: nn::ConvConfig = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };

        // Create the convolutional layers
        let conv_1: Conv3D = nn::conv3d(variable_store, in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_2: Conv3D = nn::conv3d(variable_store, in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_3: Conv3D = nn::conv3d(variable_store, in_channels, 32, KERNEL_SIZE, conv_config);
        let conv_4: Conv3D = nn::conv3d(variable_store, 32, 64, KERNEL_SIZE, conv_config);
        let conv_5: Conv3D = nn::conv3d(variable_store, 64, 128, KERNEL_SIZE, conv_config);
        let conv_6: Conv3D = nn::conv3d(variable_store, 128, 256, KERNEL_SIZE, conv_config);

        // Create the linear layer
        let fc: Linear = nn::linear(variable_store, 256, 1, Default::default());

        // Return the network
        Self { conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, fc}
    }
}
impl ModuleT for Net3D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual: &Tensor = xs;

        // Apply the first residual block
        let xs: &Tensor = &xs
            .apply(&self.conv_1)
            .relu()
            .apply(&self.conv_2);
        let xs: Tensor = (xs + residual).relu();

        // Apply another convolutional layer, this time without a residual block
        let xs: Tensor = xs
            .apply(&self.conv_3)
            .relu()
            .apply(&self.conv_4)
            .relu()
            .apply(&self.conv_5)
            .relu()
            .apply(&self.conv_6);

        // Apply GAP
        let xs: Tensor = xs
            .adaptive_avg_pool3d(&[1, 1, 1])
            .view([xs.size()[0], -1]);

        // Convert to scalar
        let xs: Tensor = xs.apply(&self.fc);

        // Apply the final activation function
        xs.tanh()
    }
}

pub enum ChessNet {
    Net2D(Net2D),
    Net3D(Net3D)
}


impl ChessNet {
    pub fn new(vs: &nn::Path, dimensions: Vec<usize>) -> Self {
        // Get the number of channel dimensions
        let channel_dimensions: i64 = dimensions[0] as i64;

        // Get the number of spatial dimensions
        let spatial_dimensions: usize = dimensions.len() - 1;

        match spatial_dimensions {
            2 => ChessNet::Net2D(Net2D::new(vs, channel_dimensions)),
            3 => ChessNet::Net3D(Net3D::new(vs, channel_dimensions)),
            _ => panic!("Bow to me, mere mortals, for I have achieved the impossible case")
        }
    }

    pub fn evaluate(&self, state_tensor: Tensor) -> f64 {
        let evaluated_tensor: Tensor = match self {
            ChessNet::Net2D(network) => network.forward_t(&state_tensor, false),
            ChessNet::Net3D(network) => network.forward_t(&state_tensor, false)
        };
        evaluated_tensor.double_value(&[0])
    }

    pub fn train(
        &self,
        variable_store: &nn::VarStore,
        matches: i64,
        base_game: Game,
        matches_folder: Option<&Path>
    ) {
        // Create the optimizer
        let mut optimizer: Optimizer = Adam::default().build(variable_store, LAMBDA).unwrap();

        // Play a certain amount of matches
        for m in 1..=matches {
            // Initialize the game
            let mut game: Game = base_game.clone();

            // Keep track of the number of moves
            let mut move_number: usize = 0;

            // Save the state tensors into a vector
            let mut state_tensors: Vec<Vec<Tensor>> = Vec::new();
            let mut predictions: Vec<Vec<Tensor>> = Vec::new();
            for _ in 0..game.number_of_players() {
                state_tensors.push(Vec::new());
                predictions.push(Vec::new());
            }

            // Play until checkmate or draw has been reached
            while !game.is_over() {
                // Get the players
                let players: &Vec<Player> = game.get_players();

                // Get the number of players
                let n_players: usize = game.number_of_players();

                // Get the current player
                let current_player_index: usize = move_number % n_players;
                let current_player: &Player = &players[current_player_index];

                if !current_player.is_in_the_game() { move_number += 1; continue; }

                // Save the state tensor
                state_tensors[current_player_index].push(game.state_tensor(move_number));

                // Get all the possible moves for the current player
                let possible_moves: Vec<ChessBoard> = game.possible_moves_for_player(current_player);

                // Rate each move
                let move_values: Vec<Tensor> = possible_moves
                    .iter()
                    .map(
                        |x| {
                            // Make the move
                            game.new_move(x.clone());

                            // Get the state tensor
                            let state_tensor: Tensor = game.state_tensor(move_number + 1);

                            // Step back
                            game.unmake_move();

                            // Rate the position
                            match self {
                                ChessNet::Net2D(network) => network.forward_t(&state_tensor, true),
                                ChessNet::Net3D(network) => network.forward_t(&state_tensor, true)
                            }
                        }
                    ).collect();

                // Select the best performing move
                let (best_move, rating): (ChessBoard, Tensor) = zip(possible_moves, move_values)
                    .into_iter()
                    .max_by(|(_, a), (_, b)| {
                        let x: f64 = a.double_value(&[0]);
                        let y: f64 = b.double_value(&[0]);

                        x.total_cmp(&y)
                    })
                    .unwrap();

                // Save the rating
                predictions[current_player_index].push(rating);

                // Make the best move
                game.new_move(best_move);

                // Increase the move counter
                move_number += 1;
            }

            // Backpropagate
            for i in 0..move_number {
                // Get the player index
                let player_index: usize = i % game.number_of_players();

                // Get the index of the object inside the vector
                let vector_index: usize = i / game.number_of_players();

                // Get the prediction
                let y: &Tensor = &predictions[player_index][vector_index];

                // Get the next prediction
                let next_player_index: usize = player_index + 1;
                let next_vector_index: usize = vector_index + (player_index == game.number_of_players()) as usize;
                let next_player_index: usize = next_player_index % game.number_of_players();

                let y_2: &Tensor = &-&predictions[next_player_index][next_vector_index];

                // Get the result of the game
                let result: f64 = if game.winner() == player_index { 1.0 } else { -1.0 };
                let y_hat: Tensor = Tensor::from(result);

                // Calculate bootstrapping loss
                let loss_bs: Tensor = (y - y_2).pow_tensor_scalar(2);

                // Calculate Monte Carlo loss
                let loss_mc: Tensor = (y - y_hat).pow_tensor_scalar(2);

                // Calculate the final loss
                let loss: Tensor = ALPHA * loss_bs + (1.0 - ALPHA) * loss_mc;

                // Step backwards
                optimizer.backward_step(&loss);
            }

            // Save the match
            match matches_folder {
                Some(path) => {
                    // Get the ID of the match
                    let match_id: usize = read_dir(path).unwrap().count() + 1;

                    // Get the complete path
                    let file_path: PathBuf = path.join(format!("{match_id}.json"));

                    match game.export(&file_path) {
                        Ok(_) => (),
                        Err(e) => println!("Failed to save match {match_id}.")
                    }
                },
                None => ()
            }
        }
    }
}

pub fn build_model(game: &Game) -> ChessNet {
    // Get the device
    let device: Device = Device::cuda_if_available();

    // Create the variable storage
    let vs: nn::VarStore = nn::VarStore::new(device);

    // Get the dimensions
    let dimensions: Vec<usize> = game.dim().clone();

    // Create the model
    let model: ChessNet = ChessNet::new(
        &vs.root(), dimensions
    );

    model
}