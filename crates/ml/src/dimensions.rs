pub type Dimensions = Vec<usize>;


#[macro_export]
macro_rules! dim {
    ( $( $x:expr ),+ ) => {
        vec![ $( $x ),+ ]
    };
}
