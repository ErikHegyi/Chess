use crate::{Layer, Tensor, ALPHA, EPSILON};

pub struct CNN {
    layers: Vec<Layer>,
    beta_1: f64,
    beta_2: f64,
    lambda: f64
}

impl CNN {
    pub fn new(
        layers: Vec<Layer>,
        beta_1: f64,
        beta_2: f64,
        learning_rate: f64
    ) -> Self {
        Self { layers, beta_1, beta_2, lambda: learning_rate }
    }
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        let mut a: Tensor = x;
        for layer in &mut self.layers {
            a = layer.forward(a);
        }
        a
    }
    pub fn backpropagate_and_step(&mut self, y: f64, y_2: f64, y_hat: f64) { 
        // Get the number of layers
        let n_layers: usize = self.layers.len();
        assert!(n_layers > 2);
        
        let y_derivative: f64 = self.y_derivative(y, y_2, y_hat);
        let u_derivative: f64 = y_derivative * (1.0 - y).powi(2);
        for i in 1..=n_layers {
            let index: usize = n_layers - i;
            
            // Select the layer
            let layer: Layer = self.layers[index];
            
            // Get the layer's inputs
            
            
        }
    }
    fn loss_monte_carlo(&self, y: f64, y_hat: f64) -> f64 { (y - y_hat).powi(2) }
    fn loss_bootstrapping(&self, y_1: f64, y_2: f64) -> f64 { (y_1 - y_2).powi(2) }
    pub fn loss(&self, y: f64, y_2: f64, y_hat: f64) -> f64 {
        let loss_mc: f64 = self.loss_monte_carlo(y, y_hat);
        let loss_bs: f64 = self.loss_bootstrapping(y, y_2);
        ALPHA * loss_bs + (1.0 - ALPHA) * loss_mc
    }
    fn y_derivative(&self, y: f64, y_2: f64, y_hat: f64) -> f64 {
        2.0 * ALPHA * (y - y_2) + 2.0 * (1.0 - ALPHA) * (y - y_hat)
    }
    
}