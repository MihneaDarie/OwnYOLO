use crate::graph_form::nodes::node::Node;
use anyhow::{Ok, Result};

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
pub struct ConvNode {
    auto_pad: AutoPad,
    kernel_shape: [i64; 2],
    group: i64,
    pads: [i64; 4],
    strides: [i64; 2],
    dilations: [i64; 2],

    next_node: Option<Box<dyn Node>>,
}

impl ConvNode {
    pub fn new(
        auto_pad: &str,
        kernel_shape: [i64; 2],
        group: i64,
        pads: [i64; 4],
        strides: [i64; 2],
        dilations: [i64; 2],
    ) -> Self {
        Self {
            auto_pad: AutoPad::from_str(auto_pad),
            kernel_shape,
            group,
            pads,
            strides,
            dilations,
            next_node: None,
        }
    }
}

impl Node for ConvNode {
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
