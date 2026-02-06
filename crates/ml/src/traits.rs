/// # Trait Mean
/// Calculate the mean average of the tensor
pub trait Mean {
    fn mean(&self) -> f64;
}

/// # Trait Median
/// Calculate the median average of the tensor
pub trait Median {
    fn median(&self) -> f64;
}

/// # Trait Minimum
/// Find and return the smallest value in the tensor
pub trait Minimum {
    fn min(&self) -> f64;
}

/// # Trait Minimum Index
/// Find the smallest value in the tensor, and return its index
pub trait MinimumIndex {
    fn min_index(&self) -> usize;
}

/// # Trait Absolute Minimum
/// Find the value, which is closest to zero.
/// In other words, find the value, which has the smallest absolute value.
/// ## Important
/// This function finds the smallest value by absolute value, but it returns the original value,
/// meaning that it can return a negative value.
pub trait AbsMinimum {
    fn abs_min(&self) -> f64;
}

/// # Trait Absolute Minimum Index
/// Find the value, which is closest to zero, and return its index.
/// In other words, find the value, which has the smallest absolute value, and return its index.
pub trait AbsMinimumIndex {
    fn abs_min_index(&self) -> usize;
}

/// # Trait Maximum
/// Find and return the largest value in the tensor
pub trait Maximum {
    fn max(&self) -> f64;
}

/// # Trait Minimum Index
/// Find the largest value in the tensor, and return its index
pub trait MaximumIndex {
    fn max_index(&self) -> usize;
}

/// # Trait Absolute Maximum
/// Find the value, which is farthest from zero.
/// In other words, find the value, which has the largest absolute value.
/// ## Important
/// This function finds the largest value by absolute value, but it returns the original value,
/// meaning that it can return a negative value.
pub trait AbsMaximum {
    fn abs_max(&self) -> f64;
}

/// # Trait Absolute Maximum Index
/// Find the value, which is farthest from zero, and return its index.
/// In other words, find the value, which has the largest absolute value, and return its index.
pub trait AbsMaximumIndex {
    fn abs_min_index(&self) -> usize;
}

/// # Trait Keep Minimum
/// Iterate through each element in the tensor, and check whether the element is smaller than the
/// provided value. If it is smaller, keep it, if it is larger, rewrite it with the
/// provided value.
pub trait KeepMinimum {
    fn keep_min(&mut self, min: f64) -> ();
}

/// # Trait Keep Maximum
/// Iterate through each element in the tensor, and check whether the element is larger than the
/// provided value. If it is larger, keep it, if it is smaller, rewrite it with the
/// provided value.
pub trait KeepMaximum {
    fn keep_max(&mut self, max: f64) -> ();
}

/// # Trait Clamp
/// Iterate through each element in the tensor, and check the value of the element.
/// If the element is smaller than the provided minimum, rewrite it with the provided minimum.
/// If the element is between the provided minimum and maximum values, keep it.
/// If the element is greater than the provided maximum, rewrite it with the provided maximum.
pub trait Clamp {
    fn clamp(&mut self, min: f64, max: f64) -> ();
}

/// # Trait Dimensions
/// Get the dimensions of the tensor.
pub trait Dimensions {
    fn dim(&self) -> Vec<usize>;
}
