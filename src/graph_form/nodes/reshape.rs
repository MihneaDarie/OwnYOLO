use std::collections::HashMap;

use crate::graph_form::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ReshapeNode<T: Default> {
    data: String,
    shape: String,

    o: String,

    unique_id: UniqueId,

    allow_zero: bool,
    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> FromHashMap for ReshapeNode<T> {
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
            unique_id: UniqueId::Reshape,
            next_node: None,
        })
    }
}

impl<T: Default> ReshapeNode<T> {
    pub fn new(allow_zero: bool) -> Self {
        Self {
            data: String::new(),
            shape: String::new(),

            o: String::new(),
            unique_id: UniqueId::Reshape,
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

impl<T: Default> Node<T> for ReshapeNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let [data, shape, result] = omap.get_disjoint_mut([&self.data, &self.shape, &self.o]);
        let data = &*data.unwrap();
        let shape = &*shape.unwrap();

        match result {
            Some(result) => {
                data.reshape(&shape, self.allow_zero, result).unwrap();
            }
            _ => panic!(
                "ReshapeNode: missing input(s) - data={} shape={}",
                self.data, self.shape
            ),
        }

        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        println!("reshape-{},{},{}", self.data, self.shape, self.o);
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
