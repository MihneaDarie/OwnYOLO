use std::collections::HashMap;

use crate::graph_form::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;

#[derive(Default)]
pub struct SliceNode<T: Default> {
    data: String,
    starts: String,
    ends: String,
    axes: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> SliceNode<T> {
    pub fn new() -> Self {
        Self {
            data: String::new(),
            starts: String::new(),
            ends: String::new(),
            axes: String::new(),
            o: String::new(),
            unique_id: UniqueId::Slice,
            next_node: None,
        }
    }
    pub fn add_input_strings(&mut self, data: String, starts: String, ends: String, axes: String) {
        self.data = data;
        self.starts = starts;
        self.ends = ends;
        self.axes = axes;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for SliceNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let [data, starts, ends, axes, o] =
            omap.get_disjoint_mut([&self.data, &self.starts, &self.ends, &self.axes, &self.o]);
        let data = &*data.unwrap();
        let starts = &*starts.unwrap();
        let ends = &*ends.unwrap();
        let axes = &*axes.unwrap();

        match (o) {
            Some(result) => {
                data.slice(&starts, &ends, &axes, result).unwrap();
            }
            _ => panic!(
                "SliceNode: missing input(s) - data={} starts={} ends={} axes={}",
                self.data, self.starts, self.ends, self.axes
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
        println!(
            "slice-{},{},{},{},{}",
            self.data, self.starts, self.ends, self.axes, self.o
        );
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
