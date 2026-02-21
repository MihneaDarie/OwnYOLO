use std::collections::HashMap;

use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct SplitNode {
    input: String,
    split: String,

    o: String,

    axis: i64,
    num_outputs: i64,

    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for SplitNode {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            input: String::new(),
            split: String::new(),

            o: String::new(),

            axis: match attrs.get("axis") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            num_outputs: match attrs.get("num_outputs") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            next_node: None,
        })
    }
}

impl SplitNode {
    pub fn new(axis: i64, num_outputs: i64) -> Self {
        Self {
            input: String::new(),
            split: String::new(),

            o: String::new(),
            axis,
            num_outputs,
            next_node: None,
        }
    }

     pub fn add_input_strings(
        &mut self,
        input: String,
        split: String,
    ) {
        self.input = input;
        self.split = split;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }

}

impl Node for SplitNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!("split-{},{}", self.axis, self.num_outputs);
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
