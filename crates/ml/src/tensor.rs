use std::fmt::{Debug, Display, Formatter};

pub struct Tensor {
    data: Vec<f64>,
    dimensions: Vec<usize>
}


impl Tensor {
    pub fn new(data: Vec<f64>, dimensions: Vec<usize>) -> Self { todo!() }

    fn strides(&self) -> Vec<usize> { todo!() }

    /// Return the dimensions of the tensor as a vector of `usize`.
    /// Each element represents the size of the corresponding dimension.
    /// Example: a 2x3 matrix would return `[2, 3]`.
    pub fn dim(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    /// Remove all dimensions of size 1 from the tensor.
    /// Example: a tensor with shape `[1, 3, 1, 4]` becomes `[3, 4]`.
    /// If all dimensions are 1, returns `[1]`.
    pub fn squeeze(&self) -> Self { todo!() }

    /// Add a new dimension of size 1 at the end of the tensor.
    /// Example: a tensor of shape `[3, 4]` becomes `[3, 4, 1]`
    pub fn unsqueeze(&self) -> Self { todo!() }

    /// Flatten the tensor into a 1D tensor (vector) while preserving all data.
    /// Example: a `[2, 3]` tensor becomes `[6]`.
    pub fn flatten(&self) -> Self {
        Self {
            data: self.data.clone(),
            dimensions: vec![self.data.len()]
        }
    }

    /// Compute the mean (average) of all elements in the tensor.
    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / (self.data.len() as f64)
    }

    /// Return the smallest element in the tensor.
    pub fn min(&self) -> f64 {
        *self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Return the index of the smallest element in the flat data array.
    /// Useful for finding the position of minimum value.
    pub fn min_index(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0
    }

    /// Return the smallest absolute value in the tensor.
    /// Computes `min(|x_i|)` over all elements.
    pub fn abs_min(&self) -> f64 {
        *self.data
            .iter()
            .min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
    }

    /// Return the index of the smallest absolute value in the tensor.
    pub fn abs_min_index(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap().0
    }

    /// Return the largest element in the tensor.
    pub fn max(&self) -> f64 {
        *self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Return the index of the largest element in the flat data array.
    pub fn max_index(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0
    }

    /// Return the largest absolute value in the tensor.
    pub fn abs_max(&self) -> f64 {
        *self.data
            .iter()
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
    }

    /// Return the index of the largest absolute value in the tensor.
    pub fn abs_max_index(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap().0
    }

    /// Return a new tensor where elements larger than `x` are replaced with `x`.
    /// Keeps smaller elements unchanged.
    pub fn keep_smaller(&self, x: f64) -> Self {
        let new_data: Vec<f64> = self.data
            .iter()
            .map(|v| if *v < x { *v } else { x })
            .collect();
        Self::new(new_data, self.dimensions.clone())
    }

    /// Return a new tensor where elements smaller than `x` are replaced with `x`.
    /// Keeps larger elements unchanged.
    pub fn keep_larger(&self, x: f64) -> Self {
        let new_data: Vec<f64> = self.data
            .iter()
            .map(|v| if *v > x { *v } else { x })
            .collect();
        Self::new(new_data, self.dimensions.clone())
    }

    /// Clamp each element to be within `[min, max]`.
    /// Elements smaller than `min` become `min`, elements larger than `max` become `max`.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let new_data: Vec<f64> = self.data
            .iter()
            .map(|v| v.min(max).max(min))
            .collect();
        Self::new(new_data, self.dimensions.clone())
    }

    /// If the tensor has only one element, return it.
    /// Panics if the tensor has more than one element.
    pub fn item(&self) -> f64 {
        if self.data.len() != 1 {
            panic!("Tensor has more than one element")
        } else {
            self.data[0]
        }
    }
}

impl Display for Tensor {  // text: Tensor of size insert dimensions
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Debug for Tensor {  // Display the tensor
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl From<Vec<f64>> for Tensor {  // Create a 1D tensor
    fn from(value: Vec<f64>) -> Self {
        todo!()
    }
}

impl From<Vec<Vec<f64>>> for Tensor {  // Create a 2D tensor
    fn from(value: Vec<Vec<f64>>) -> Self {
        todo!()
    }
}

// TODO: Create a macro for creating an N-dimensional tensor using square brackets, like this
/*
let vector = tensor![
    1, 2, 3, 4, 5
];
let matrix = tensor![
    [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
];
let tensor = tensor![
    [
        [
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
        ],
        [
            [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
        ]
    ],
    [
        [
            [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]
        ],
        [
            [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]
        ]
    ],
    [
        [
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
        ],
        [
            [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
        ]
    ],
    [
        [
            [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]
        ],
        [
            [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]
        ]
    ]
]
*/