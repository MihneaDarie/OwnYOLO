#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UniqueId {
    Add,
    Concat,
    Conv,
    Div,
    MaxPool,
    Mul,
    Reshape,
    Resize,
    Sigmoid,
    Slice,
    Softmax,
    Split,
    Sub,
    Transpose,
    #[default]
    Undefined,
}
