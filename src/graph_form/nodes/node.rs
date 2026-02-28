use std::collections::HashMap;

use anyhow::{Ok, Result};

use crate::graph_form::{
    nodes::{sigmoid::SigmoidNode, silu::SiluNode, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

pub fn fuse_silu() {}

pub trait Node<T: Default + 'static> {
    fn pass(&self, omap: &mut TensorMap) {}
    fn print(&self);
    fn self_count(&self, count: usize) -> usize;
    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()>;
    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>>;

    fn get_next_mut(&mut self) -> Option<&mut Vec<Box<dyn Node<T>>>>;
    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>);
    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>>;

    fn input_names(&self) -> Vec<String>;
    fn output_names(&self) -> Vec<String>;
    fn get_unique_id(&self) -> UniqueId;
    fn get_unique_id_mut(&mut self) -> UniqueId;

    fn optimize_further(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
