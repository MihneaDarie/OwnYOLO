use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct TransposeNode {
    perm: Vec<i64>,

    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for TransposeNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            perm: match attrs.get("perm") {
                Some(av) => av.as_ints().unwrap().to_vec(),
                None => vec![],
            },
            next_node: None,
        })
    }
}

impl TransposeNode {
    pub fn new(perm: Vec<i64>) -> Self {
        Self {
            perm,
            next_node: None,
        }
    }
}

impl Node for TransposeNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("transpose-{:?}", self.perm);
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
