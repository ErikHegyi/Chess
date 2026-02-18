use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Index, Mul, Sub};


#[derive(Clone)]
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
        if self.data.len() > 1 {
            panic!("Tensor has more than one element")
        } else if self.data.len() < 1 {
            panic!("Tensor does not have any elements")
        } else {
            self.data[0]
        }
    }

    pub fn zeros(dimensions: Vec<usize>) -> Self {
        Self {
            data: vec![0.0; dimensions.iter().product()],
            dimensions
        }
    }

    pub fn wrap_scalar(x: f64) -> Self {
        todo!()
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum::<f64>()
    }

    pub fn transpose(&self) -> Self { todo!() }
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

impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = &'a Tensor;

    fn mul(self, rhs: &'a Tensor) -> Self::Output {
        todo!()
    }
}

impl<'a> Mul<f64> for &'a Tensor {
    type Output = &'a Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        todo!()
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        todo!()
    }
}

impl Add<Tensor> for Tensor {
    type Output = Self;
    fn add(self, rhs: Tensor) -> Self::Output {
        todo!()
    }
}

impl<'a> Add<&'a Tensor> for Tensor {
    type Output = &'a Tensor;
    fn add(self, rhs: &'a Tensor) -> Self::Output {
        todo!()
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        todo!()
    }
}

impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = &'a Tensor;
    fn sub(self, rhs: &'a Tensor) -> Self::Output {
        todo!()
    }
}

impl<'a> Sub<&'a Tensor> for &'a Tensor {
    type Output = &'a Tensor;
    fn sub(self, rhs: &'a Tensor) -> Self::Output {
        todo!()
    }
}

impl<'a> Add<f64> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        // Create a new vector
        let dimensions: Vec<usize> = self.dimensions.clone();
        let data: Vec<f64> = self.data.iter().map(|x| x + rhs).collect::<Vec<f64>>();
        Tensor::new(data, dimensions)
    }
}

impl<'a> Sub<f64> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Self::Output {
        // Create a new vector
        let dimensions: Vec<usize> = self.dimensions.clone();
        let data: Vec<f64> = self.data.iter().map(|x| x - rhs).collect::<Vec<f64>>();
        Tensor::new(data, dimensions)
    }
}

impl Index<usize> for Tensor {
    type Output = Tensor;
    fn index(&self, index: usize) -> &Self::Output {
        todo!()
    }
}