use std::collections::HashMap;

use crate::graph_form::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct SoftMaxNode<T: Default> {
    input: String,

    o: String,

    unique_id: UniqueId,

    axis: i64,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> FromHashMap for SoftMaxNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            input: String::new(),
            o: String::new(),
            axis: match attrs.get("axis") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            unique_id: UniqueId::Softmax,
            next_node: None,
        })
    }
}

impl<T: Default> SoftMaxNode<T> {
    pub fn new(axis: i64) -> Self {
        Self {
            input: String::new(),

            o: String::new(),

            unique_id: UniqueId::Softmax,

            axis,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, input: String) {
        self.input = input;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for SoftMaxNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.input, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                x.softmax(self.axis, result).unwrap();
            }
            None => panic!("SoftMaxNode: missing input {}", self.input),
        }
        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        println!("soft_max-{},{}", self.input, self.o);
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

    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()> {
        if let Some(next_node) = &mut self.next_node {
            next_node.insert(next)?;
            return Ok(());
        } else {
            self.next_node = Some(next)
        }
        Ok(())
    }
}
