use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub enum BoardMask {
    BoardMask2D(Vec<Vec<bool>>),
    BoardMask3D(Vec<Vec<Vec<bool>>>)
}