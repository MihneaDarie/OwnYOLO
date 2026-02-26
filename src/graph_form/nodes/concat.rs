use std::collections::HashMap;

use crate::graph_form::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ConcatNode<T> {
    inputs: Vec<String>,

    o: String,

    unique_id: UniqueId,

    axis: i64,
    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> ConcatNode<T> {
    pub fn add_input_strings(&mut self, inputs: Vec<String>) {
        self.inputs = inputs;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> FromHashMap for ConcatNode<T> {
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
            unique_id: UniqueId::Concat,
        })
    }
}

impl<T: Default> Node<T> for ConcatNode<T> {
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
        let arrays: Vec<&TypedArray> = self
            .inputs
            .iter()
            .map(|name| {
                omap.get(name)
                    .unwrap_or_else(|| panic!("ConcatNode: missing input {}", name))
            })
            .collect();

        let ndim = match &arrays[0] {
            TypedArray::F32(a) => a.ndim(),
            TypedArray::F64(a) => a.ndim(),
            TypedArray::I32(a) => a.ndim(),
            TypedArray::I64(a) => a.ndim(),
            _ => panic!("unsupported type in concat"),
        };

        let axis = if self.axis < 0 {
            (ndim as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let refs: Vec<&TypedArray> = arrays;
        let mut result = TypedArray::Undefined;
        TypedArray::concat(&refs, axis, &mut result).unwrap();
        omap.insert(self.o.clone(), result);

        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn print(&self) {
        println!("concat-{:?},{}", self.inputs, self.o);
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
