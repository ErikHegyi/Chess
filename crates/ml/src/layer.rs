use crate::{ActivationFunction, Tensor, Dimensions, ALPHA};

pub struct Layer {
    // Define the parameters
    weights: Tensor,
    bias: f64,

    // Define the activation function
    activation_function: ActivationFunction,

    // Input and output shapes
    input_dimensions: Dimensions,
    output_dimensions: Dimensions,

    // Keep track of the data acquired during the forward pass
    x: Tensor,  // Input
    u: Tensor,  // Output before activation function
    y: Tensor,  // Output after activation function + residual re-adding if necessary
    eta: bool  // Keep track of whether the layer is residual or not
}


impl Layer {
    pub fn new(
        input_dimensions: Dimensions,
        output_dimensions: Dimensions,
        kernel_size: Dimensions,
        is_residual_block: bool,
        activation_function: ActivationFunction
    ) -> Self {
        // Residual blocks can not change dimensions
        if is_residual_block {
            assert_eq!(input_dimensions, output_dimensions)
        }

        // Create the layer
        Self {
            weights: Tensor::zeros(kernel_size),
            bias: 0.0,
            activation_function,
            input_dimensions: input_dimensions.clone(),
            output_dimensions: output_dimensions.clone(),
            x: Tensor::zeros(input_dimensions),
            u: Tensor::zeros(output_dimensions.clone()),
            y: Tensor::zeros(output_dimensions),
            eta: is_residual_block
        }
    }

    pub fn reset_gradients(&mut self) { todo!() }

    pub fn forward(&mut self, x: Tensor) -> Tensor {
        // Save the input
        self.x = x.clone();
        
        // Calculate the output before activation
        let z: Tensor = &self.weights * &x + self.bias;

        // Save the value
        self.u = z.clone();

        // Apply the activation function
        let mut y: Tensor = match self.activation_function {
            // Rectified Linear Unit
            ActivationFunction::ReLU => z.keep_larger(0.0),

            // Global Average Pooling
            ActivationFunction::GAP => {
                // The dimensions of the input and the layer's input must be the same
                assert_eq!(x.dim()[0], self.input_dimensions[0]);

                // Calculate the spatial dimension sizes
                let m: usize = self.input_dimensions[1..].iter().product();

                // Get the number of channel dimensions
                let dim: usize = self.input_dimensions[0];

                // Save the values into a vector
                let mut values: Vec<f64> = vec![0.0; dim];
                for d in 0..dim {
                    // Average out the values in the spatial dimensions
                    values[d] = x[d].sum() / (m as f64)
                }

                // Create a new one-dimensional tensor
                Tensor::new(values, vec![dim])
            },

            // Hyperbolic Tangent
            ActivationFunction::TanH => Tensor::wrap_scalar(z.item().tanh())
        };

        // If the layer is residual, add the original value back
        if self.eta {
            y = y + x;
        }

        // Save the value
        self.y = y.clone();

        // Return the value
        y
    }
}