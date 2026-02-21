use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ReshapeNode {
    data: String,
    shape: String,

    o: String,

    allow_zero: bool,
    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for ReshapeNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            data: String::new(),
            shape: String::new(),

            o: String::new(),
            allow_zero: {
                match attrs.get("allow_zero") {
                    Some(av) => av.as_int().unwrap() != 0,
                    None => false,
                }
            },
            next_node: None,
        })
    }
}

impl ReshapeNode {
    pub fn new(allow_zero: bool) -> Self {
        Self {
            data: String::new(),
            shape: String::new(),

            o: String::new(),
            allow_zero,
            next_node: None,
        }
    }
     pub fn add_input_strings(&mut self, data: String, shape: String) {
        self.shape = shape;
        self.data = data;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl Node for ReshapeNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("rehape-{}", self.allow_zero);
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
