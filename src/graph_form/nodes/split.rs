use std::collections::HashMap;

use crate::graph_form::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct SplitNode<T: Default> {
    input: String,
    split: String,

    o: Vec<String>,

    unique_id: UniqueId,

    axis: i64,
    num_outputs: i64,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> FromHashMap for SplitNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            input: String::new(),
            split: String::new(),

            o: vec![],

            unique_id: UniqueId::Split,

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

impl<T: Default> SplitNode<T> {
    pub fn new(axis: i64, num_outputs: i64) -> Self {
        Self {
            input: String::new(),
            split: String::new(),

            o: vec![],
            axis,
            num_outputs,
            unique_id: UniqueId::Split,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, input: String, split: String) {
        self.input = input;
        self.split = split;
    }

    pub fn add_output_strings(&mut self, o: Vec<String>) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for SplitNode<T> {
    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Box<dyn Node<T>>> {
        self.next_node.as_ref()
    }

    fn pass(&self, omap: &mut TensorMap) {
        let input = omap.get(&self.input);

        let split_sizes: Vec<i64> = if let Some(TypedArray::I64(a)) = omap.get(&self.split) {
            a.iter().cloned().collect()
        } else if self.num_outputs > 0 {
            let input_ref = input.as_ref().unwrap();
            let axis = self.axis as usize;
            let dim = match input_ref {
                TypedArray::F32(a) => a.shape()[axis],
                _ => panic!("unsupported type"),
            };
            let chunk = dim / self.num_outputs as usize;
            vec![chunk as i64; self.num_outputs as usize]
        } else {
            panic!("SplitNode: no split tensor and no num_outputs");
        };

        match input {
            Some(input) => {
                let split_tensor = TypedArray::I64(ndarray::Array1::from(split_sizes).into_dyn());
                let mut results = Vec::new();
                input.split(&split_tensor, self.axis, &mut results).unwrap();

                for (name, chunk) in self.o.iter().zip(results.into_iter()) {
                    omap.insert(name.clone(), chunk);
                }
            }
            None => panic!("SplitNode: missing input {}", self.input),
        }
        
        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn output_names(&self) -> Vec<String> {
        self.o.clone()
    }

    fn print(&self) {
        println!("split-{},{},{:?}", self.input, self.split, self.o);
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
