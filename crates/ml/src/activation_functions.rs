pub enum ActivationFunction {
    ReLU,
    GAP,
    TanH
}


impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{function_name}", function_name = match *self {
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::GAP => "GAP",
            ActivationFunction::TanH => "tanh"
        })
    }
}
