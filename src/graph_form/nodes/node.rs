use std::collections::HashMap;

use anyhow::{Ok, Result};

use crate::graph_form::{
    nodes::{sigmoid::SigmoidNode, silu::SiluNode, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

pub trait Node<T: Default + 'static> {
    fn pass(&self, omap: &mut TensorMap) {}
    fn print(&self);
    fn self_count(&self, count: usize) -> usize;
    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()>;
    fn get_next(&self) -> Option<Vec<&Box<dyn Node<T>>>>;

    fn get_next_mut(&mut self) -> Option<Vec<&mut Box<dyn Node<T>>>>;
    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>);
    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>>;

    fn input_names(&self) -> Vec<String>;
    fn output_names(&self) -> Vec<String>;
    fn get_unique_id(&self) -> UniqueId;
    fn get_unique_id_mut(&mut self) -> UniqueId;

    fn optimize_further(&mut self) -> anyhow::Result<()> {
        // let should_fuse = {
        //     if let Some(next) = self.get_next() {
        //         if let Some(next_next) = next.get_next() {
        //             next.get_unique_id() == UniqueId::Sigmoid
        //                 && next_next.get_unique_id() == UniqueId::Mul
        //         } else {
        //             false
        //         }
        //     } else {
        //         false
        //     }
        // };

        // if should_fuse && self.get_unique_id() == UniqueId::Conv {
        //     let mut silu_node = SiluNode::<T>::new();

        //     let x_name = self.output_names()[0].clone();
        //     let o_name;
        //     let tail;

        //     {
        //         let next = self.get_next_mut().unwrap();
        //         let next_next = next.get_next_mut().unwrap();
        //         o_name = next_next.output_names()[0].clone();
        //         tail = next_next.take_next();
        //     }

        //     silu_node.x = x_name;
        //     silu_node.o = o_name;
        //     silu_node.next_node = tail;

        //     self.set_next(Some(Box::new(silu_node)));
        // }

        // if let Some(next) = self.get_next_mut() {
        //     next.optimize_further()?;
        // }

        Ok(())
    }
}
