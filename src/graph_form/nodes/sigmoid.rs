use std::collections::HashMap;

use crate::graph_form::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;

#[derive(Default)]
pub struct SigmoidNode<T: Default> {
    x: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> SigmoidNode<T> {
    pub fn new() -> Self {
        Self {
            x: String::new(),
            o: String::new(),
            unique_id: UniqueId::Sigmoid,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for SigmoidNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                x.sigmoid(result).unwrap();
            }
            None => panic!("SigmoidNode: missing input {}", self.x),
        }
        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        println!("sigmoid-{},{}", self.x, self.o);
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
