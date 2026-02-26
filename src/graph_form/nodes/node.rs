use std::collections::HashMap;

use anyhow::Result;

use crate::graph_form::{
    nodes::unique_ids::UniqueId, tensor_map::TensorMap, typed_array::TypedArray,
};

pub trait Node<T: Default> {
    fn pass(&self, omap: &mut TensorMap) {}
    fn print(&self);
    fn self_count(&self, count: usize) -> usize;
    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()>;
    fn get_next(&self) -> Option<&Box<dyn Node<T>>>;
    fn output_names(&self) -> Vec<String>;
    fn get_unique_id(&self) -> UniqueId;
}
