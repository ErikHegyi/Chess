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
use crate::{hyperparameters::*, Export, Variant};
use crate::state::{Move, StateTensor};
use rand;
use rand::RngExt;

#[derive(Debug)]
struct Net2D {
    variable_store: nn::VarStore,
    conv_1: Conv2D,
    conv_2: Conv2D,
    conv_3: Conv2D,
    conv_4: Conv2D,
    conv_5: Conv2D,
    conv_6: Conv2D,
    fc: Linear
}
impl Net2D {
    fn new(variable_store: nn::VarStore, in_channels: i64) -> Self {
        // Configure the properties of the convolutional layers
        let conv_config: nn::ConvConfig = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };

        // Create the convolutional layers
        let conv_1: Conv2D = nn::conv2d(&variable_store.root(), in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_2: Conv2D = nn::conv2d(&variable_store.root(), in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_3: Conv2D = nn::conv2d(&variable_store.root(), in_channels, 32, KERNEL_SIZE, conv_config);
        let conv_4: Conv2D = nn::conv2d(&variable_store.root(), 32, 64, KERNEL_SIZE, conv_config);
        let conv_5: Conv2D = nn::conv2d(&variable_store.root(), 64, 128, KERNEL_SIZE, conv_config);
        let conv_6: Conv2D = nn::conv2d(&variable_store.root(), 128, 256, KERNEL_SIZE, conv_config);

        // Create the linear layer
        let fc: Linear = nn::linear(&variable_store.root(), 256, 1, Default::default());

        // Return the network
        Self { variable_store, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, fc}
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
    variable_store: nn::VarStore,
    conv_1: Conv3D,
    conv_2: Conv3D,
    conv_3: Conv3D,
    conv_4: Conv3D,
    conv_5: Conv3D,
    conv_6: Conv3D,
    fc: Linear
}
impl Net3D {
    fn new(variable_store: nn::VarStore, in_channels: i64) -> Self {
        // Configure the properties of the convolutional layers
        let conv_config: nn::ConvConfig = nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };

        // Create the convolutional layers
        let conv_1: Conv3D = nn::conv3d(&variable_store.root(), in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_2: Conv3D = nn::conv3d(&variable_store.root(), in_channels, in_channels, KERNEL_SIZE, conv_config);
        let conv_3: Conv3D = nn::conv3d(&variable_store.root(), in_channels, 32, KERNEL_SIZE, conv_config);
        let conv_4: Conv3D = nn::conv3d(&variable_store.root(), 32, 64, KERNEL_SIZE, conv_config);
        let conv_5: Conv3D = nn::conv3d(&variable_store.root(), 64, 128, KERNEL_SIZE, conv_config);
        let conv_6: Conv3D = nn::conv3d(&variable_store.root(), 128, 256, KERNEL_SIZE, conv_config);

        // Create the linear layer
        let fc: Linear = nn::linear(&variable_store.root(), 256, 1, Default::default());

        // Return the network
        Self { variable_store, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, fc}
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
    pub fn new(vs: nn::VarStore, dimensions: Vec<usize>) -> Self {
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

    pub fn evaluate(&self, state_tensor: Tensor) -> f32 {
        let evaluated_tensor: Tensor = match self {
            ChessNet::Net2D(network) => network.forward_t(&state_tensor, false),
            ChessNet::Net3D(network) => network.forward_t(&state_tensor, false)
        };
        evaluated_tensor.double_value(&[]) as f32
    }

    pub fn train<G>(
        &self,
        matches: i64,
        folder: Option<&Path>
    ) where
        G: Variant + StateTensor + Export + Clone + serde::Serialize
    {
        // Get the variable store
        let variable_store: &nn::VarStore = match self {
            ChessNet::Net2D(n) => &n.variable_store,
            ChessNet::Net3D(n) => &n.variable_store
        };

        // Create the optimizer
        let mut optimizer: Optimizer = Adam::default().build(variable_store, LAMBDA).unwrap();

        // Create a random range
        let mut rng: rand::prelude::ThreadRng = rand::rng();

        // Play a certain amount of matches
        for match_id in 1..=matches {
            let mut skip_match: bool = false;

            // Initialize the game
            let mut game: G = G::new();

            // Save the state tensors into a vector
            let mut state_tensors: Vec<Vec<Tensor>> = Vec::new();

            for _ in 0..G::number_of_players() {
                state_tensors.push(Vec::new());
            }

            // Play until checkmate or draw has been reached
            while game.winner().is_none() {
                // Get the current player
                let current_player: usize = game.player_to_move();

                // Get each possible move for the player
                let possible_moves: Vec<Move> = game.legal_moves(current_player);

                // Rate each move
                let move_values: Vec<(Tensor, Tensor)> = possible_moves
                    .iter()
                    .map(
                        |x| {
                            // Make the move
                            let state_tensor: Tensor = game.state_tensor_with_move(x.clone());
                            let state_tensor: Tensor = state_tensor.unsqueeze(0);

                            // Rate the position
                            let rating: Tensor = match self {
                                ChessNet::Net2D(network) => network.forward_t(&state_tensor, false),
                                ChessNet::Net3D(network) => network.forward_t(&state_tensor, false)
                            };

                            (state_tensor, rating)
                        }
                    ).collect();

                // Select the best performing move
                let (best_move, (state_tensor, _)): (Move, (Tensor, Tensor)) = match zip(possible_moves, move_values)
                    .into_iter()
                    .max_by(|(move_x,( _, a)), (move_y, (_, b))| {
                        let x: f32 = a.double_value(&[0]) as f32
                            + rng.random_range(-0.5..0.5)
                            + MU * game.material_advantage_with_move(move_x.clone()) as f32
                            + if game.is_other_king_in_check_with_move(move_x.clone()) { KAPPA } else { 0.0 };
                        let y: f32 = b.double_value(&[0]) as f32
                            + rng.random_range(-0.5..0.5)
                            + MU * game.material_advantage_with_move(move_y.clone()) as f32
                            + if game.is_other_king_in_check_with_move(move_y.clone()) { KAPPA } else { 0.0 };

                        x.total_cmp(&y)
                    }) {
                    Some(x) => x,
                    None => {
                        // Something went catastrophically wrong, skip the match
                        skip_match = true;
                        break;
                    }
                };

                // Make the best move
                game.move_piece(best_move);

                // Save the best move's state tensor
                state_tensors[current_player].push(state_tensor);
            }

            if skip_match { continue; }

            let n_moves: f32 = game.number_of_moves() as f32;

            // Get the result of the game
            // game.winner() returns 1 if the given player won, otherwise 0 or -1
            let mut results: Vec<f32> = Vec::new();
            for player in 0..G::number_of_players() {
                match game.winner() {
                    Some(p) => {
                        // Calculate the result
                        let result: f32 = match p {
                            -1 => -1.0,
                            _ => if player == p as usize { 1.0 } else { -1.0 }
                        };
                        results.push(result - OMICRON * n_moves)
                    },
                    None => {
                        panic!("How did the while loop end?")
                    }
                }
            }
            println!("{results:?}, {}", game.winner().unwrap());

            // Backpropagate
            let epochs: usize = match results.iter().any(|x| *x > 0.0) {
                false => 1,
                true => {
                    println!("Win on match {match_id}!");
                    5
                }
            };
            for _ in 0..epochs {
                //let mut total_loss: Option<Tensor> = None;
                let mut total_loss: Option<Tensor> = {
                    // Predict
                    let y: Tensor = match self {
                        ChessNet::Net2D(network) => network.forward_t(&state_tensors[0][0], true),
                        ChessNet::Net3D(network) => network.forward_t(&state_tensors[0][0], true)
                    };
                    let y_hat: Tensor = Tensor::from(results[0]);
                    let loss_mc: Tensor = (y - y_hat).pow_tensor_scalar(2);
                    Some(loss_mc)
                };
                for (i, movement) in game.past_moves().clone().iter().enumerate() {
                    // Get the player
                    let player: usize = movement.player;

                    // Get the index of the object inside the vector
                    let vector_index: usize = i / G::number_of_players();

                    // Get the state tensors
                    let state_tensor: &Tensor = &state_tensors[player][vector_index];

                    let next_player_index: usize = player + 1;
                    let next_vector_index: usize = vector_index + (player == G::number_of_players()) as usize;
                    let next_player_index: usize = next_player_index % G::number_of_players();

                    // Predict
                    let y: Tensor = match self {
                        ChessNet::Net2D(network) => network.forward_t(state_tensor, true),
                        ChessNet::Net3D(network) => network.forward_t(state_tensor, true)
                    };
                    let y_hat: Tensor = Tensor::from(results[movement.player]);

                    // Evaluate the position at the next move
                    let y_2: Tensor = if i + 1 < game.number_of_moves() {
                        let state_tensor_2: &Tensor = &state_tensors[next_player_index][next_vector_index];
                        match self {
                            ChessNet::Net2D(network) => (-network.forward_t(state_tensor_2, true)).detach(),
                            ChessNet::Net3D(network) => (-network.forward_t(state_tensor_2, true)).detach()
                        }
                    } else { Tensor::from(results[movement.player]) };

                    // Calculate bootstrapping loss
                    let loss_bs: Tensor = (&y - &y_2).pow_tensor_scalar(2);

                    // Calculate Monte Carlo loss
                    let loss_mc: Tensor = (y - y_hat).pow_tensor_scalar(2);

                    // Calculate the final loss
                    // Monte Carlo loss is only applicable on games with two players
                    let loss: Tensor = match G::number_of_players() {
                        2 => ALPHA * loss_bs + (1.0 - ALPHA) * loss_mc,
                        _ => loss_bs
                    };

                    // Save the loss
                    if total_loss.is_none() {
                        total_loss = Some(loss);
                    } else {
                        total_loss = Some(total_loss.unwrap() + loss);
                    }
                }

                // Step backwards
                match total_loss {
                    Some(loss) => optimizer.backward_step(&loss),
                    None => panic!("This should not have been possible")
                }
            }

            // Save the match
            match folder {
                Some(path) => {
                    // Find the full path to the folder
                    let path: PathBuf = path.join(format!("{folder_name}/matches", folder_name = G::variant_name()));

                    // Get the ID of the match
                    let match_id: usize = read_dir(path.as_path()).unwrap().count() + 1;

                    // Get the complete path to the file
                    let file_path: PathBuf = path.join(format!("{match_id}.json"));

                    match game.export(&file_path) {
                        Ok(_) => (),
                        Err(_) => println!("Failed to save match {match_id}.")
                    }
                },
                None => ()
            }

            // Export the model
            match folder {
                Some(path) => {
                    let file_path: PathBuf = path.join(format!("{folder_name}/model/model.ot", folder_name = G::variant_name()));
                    let vs: &nn::VarStore = match self {
                        ChessNet::Net2D(net) => &net.variable_store,
                        ChessNet::Net3D(net) => &net.variable_store
                    };
                    match vs.save(&file_path) {
                        Ok(_) => (),
                        Err(e) => panic!("Could not save model! ({e})")
                    };
                },
                None => ()
            }
        }
    }

    pub fn import<G>(path: &Path) -> Self
    where
        G: Variant + StateTensor + Export + Clone + serde::Serialize
    {
        // Get the device
        let device: Device = Device::cuda_if_available();

        // Create an empty variable storage
        let mut vs: nn::VarStore = nn::VarStore::new(device);
        vs.set_kind(tch::Kind::Float);

        // Build model
        let model: ChessNet = ChessNet::new(vs, G::dimensions());
        match model {
            ChessNet::Net2D(mut n) => {
                n.variable_store.load(path).unwrap();
                ChessNet::Net2D(n)
            },
            ChessNet::Net3D(mut n) => {
                n.variable_store.load(path).unwrap();
                ChessNet::Net3D(n)
            }
        }
    }
}

pub fn build_model<G>() -> ChessNet
where
    G: Variant + StateTensor + Export + Clone + serde::Serialize
{
    // Get the device
    let device: Device = Device::cuda_if_available();

    // Create the variable storage
    let mut vs: nn::VarStore = nn::VarStore::new(device);
    vs.set_kind(tch::Kind::Float);

    // Get the spatial dimensions
    let dimensions: Vec<usize> = G::dimensions();

    // Create the model
    let model: ChessNet = ChessNet::new(
        vs, dimensions
    );

    model
}