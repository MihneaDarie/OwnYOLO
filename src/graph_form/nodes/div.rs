use std::collections::HashMap;

use crate::graph_form::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;

#[derive(Default)]
pub struct DivNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> DivNode<T> {
    pub fn new() -> Self {
        Self {
            a: String::new(),
            b: String::new(),
            o: String::new(),
            unique_id: UniqueId::Div,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for DivNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(out) => {
                a.div(&b, out).unwrap();
            }
            _ => panic!("DivNode: missing input(s) - a={} b={}", self.a, self.b),
        }

        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn print(&self) {
        println!("div-{},{},{}", self.a, self.b, self.o);
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
