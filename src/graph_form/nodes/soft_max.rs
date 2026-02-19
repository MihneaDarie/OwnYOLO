use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct SoftMaxNode {
    axis: i64,

    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for SoftMaxNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            axis: match attrs.get("axis") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            next_node: None,
        })
    }
}

impl SoftMaxNode {
    pub fn new(axis: i64) -> Self {
        Self {
            axis,
            next_node: None,
        }
    }
}

impl Node for SoftMaxNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("soft_max-{}", self.axis);
        if let Some(next) = &self.next_node {
            next.print();
        }
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
