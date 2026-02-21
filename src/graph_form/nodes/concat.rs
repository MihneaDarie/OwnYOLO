use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ConcatNode {

    inputs: Vec<String>,

    o: String,

    axis: i64,
    next_node: Option<Box<dyn Node>>,
}

impl ConcatNode {
    pub fn add_input_strings(&mut self, inputs: Vec<String>) {
        self.inputs = inputs; 
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o; 
    }
}

impl FromHashMap for ConcatNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            axis: {
                match attrs.get("axis") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            next_node: None,
            inputs: vec![],
            o: String::new(),
        })
    }
}

impl Node for ConcatNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("concat-{}", self.axis);
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
