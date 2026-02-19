use crate::graph_form::nodes::node::Node;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoPad {
    #[default]
    NOTSET,
    SameUpper,
    SameLower,
    VALID,
}

impl AutoPad {
    pub fn from_str(str: &str) -> Self {
        match str {
            "SAME_UPPER" => Self::SameUpper,
            "SAME_LOWER" => Self::SameLower,
            "VALID" => Self::VALID,
            _ => Self::NOTSET,
        }
    }
}

#[derive(Default)]
pub struct MaxPoolNode {
    auto_pad: AutoPad,
    ceil_mode: i64,
    kernel_shape: [i64; 2],
    dilations: [i64; 2],
    strides: [i64; 2],
    pads: [i64; 4],
    storage_order: i64,
    next_node: Option<Box<dyn Node>>,
}

impl MaxPoolNode {
    pub fn new(
        auto_pad: &str,
        ceil_mode: i64,
        kernel_shape: [i64; 2],
        dilations: [i64; 2],
        strides: [i64; 2],
        storage_order: i64,
        pads: [i64; 4],
    ) -> Self {
        Self {
            auto_pad: AutoPad::from_str(auto_pad),
            ceil_mode,
            kernel_shape,
            dilations,
            strides,
            pads,
            storage_order,
            next_node: None,
        }
    }
}

impl Node for MaxPoolNode {
    fn pass(&self) {
        todo!()
    }
    fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.next_node {
            next.self_count(count + 1)
        } else {
            count
        }
    }
    fn insert(&mut self, next: Box<dyn Node>) -> Result<()> {
        if let Some(next_node) = &mut self.next_node {
            next_node.insert(next)?;
            return Ok(());
        } else {
            self.next_node = Some(next)
        }
        Ok(())
    }
}
