use std::collections::HashMap;

use crate::{
    graph_form::{
        nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
        tensor_map::TensorMap,
        typed_array::TypedArray,
    },
    yolo::utils::Conv2D,
};
use anyhow::{Ok, Result};
use onnx_extractor::AttributeValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoPad {
    #[default]
    NOTSET,
    SameUpper,
    SameLower,
    VALID,
}

impl AutoPad {
    pub fn from_str(str: &str) -> Self {
        match str {
            "SAME_UPPER" => Self::SameUpper,
            "SAME_LOWER" => Self::SameLower,
            "VALID" => Self::VALID,
            _ => Self::NOTSET,
        }
    }
}

#[derive(Default)]
pub struct ConvNode<T: Default> {
    x: String,
    w: String,
    b: Option<String>,

    o: String,

    unique_id: UniqueId,

    auto_pad: AutoPad,
    kernel_shape: Vec<usize>,
    group: i64,
    pads: Vec<usize>,
    strides: Vec<usize>,
    dilations: Vec<usize>,

    next_node: Option<Box<dyn Node<T>>>,
}

impl<T: Default> FromHashMap for ConvNode<T> {
    fn from_hashmap(
        attrs: &std::collections::HashMap<String, AttributeValue>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            x: String::new(),
            w: String::new(),
            b: None,
            o: String::new(),
            auto_pad: {
                match attrs.get("auto_pad") {
                    Some(av) => {
                        let pad = av.as_string().unwrap();
                        AutoPad::from_str(&pad)
                    }
                    None => AutoPad::NOTSET,
                }
            },
            kernel_shape: {
                match attrs.get("kernel_shape") {
                    Some(av) => av.as_ints().unwrap().to_vec().iter().map(|&val| val as usize).collect(),
                    None => vec![],
                }
            },
            pads: {
                match attrs.get("pads") {
                    Some(av) => av.as_ints().unwrap().to_vec().iter().map(|&val| val as usize).collect(),
                    None => vec![],
                }
            },
            strides: {
                match attrs.get("strides") {
                    Some(av) => av.as_ints().unwrap().to_vec().iter().map(|&val| val as usize).collect(),
                    None => vec![],
                }
            },
            dilations: {
                match attrs.get("dilations") {
                    Some(av) => av.as_ints().unwrap().to_vec().iter().map(|&val| val as usize).collect(),
                    None => vec![],
                }
            },
            group: {
                match attrs.get("groups") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            unique_id: UniqueId::Conv,
            next_node: None,
        })
    }
}

impl<T: Default> ConvNode<T> {
    pub fn new(
        auto_pad: &str,
        kernel_shape: Vec<usize>,
        group: i64,
        pads: Vec<usize>,
        strides: Vec<usize>,
        dilations: Vec<usize>,
    ) -> Self {
        Self {
            x: String::new(),
            w: String::new(),
            b: None,
            o: String::new(),
            auto_pad: AutoPad::from_str(auto_pad),
            kernel_shape,
            group,
            pads,
            strides,
            dilations,
            unique_id: UniqueId::Conv,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, x: String, w: String, b: Option<String>) {
        self.x = x;
        self.w = w;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> Node<T> for ConvNode<T> {
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
        let def = &String::from("");
        let b = self.b.as_ref().unwrap_or(def);
        
        let [x, w, b, o] = omap.get_disjoint_mut([&self.x, &self.w, &b, &self.o]);
        let x = &*x.unwrap();
        let w = &*w.unwrap();
        let b = match b {
            Some(b) => Some(&*b),
            None => None,
        };

        match o {
            Some(result) => {
                let cfg = Conv2D {
                    pad: self.pads.first().copied().unwrap_or(0) as usize,
                    stride: self.strides.first().copied().unwrap_or(1) as usize,
                };
                x.conv(&w, b, &cfg, result, false).unwrap();
            }
            _ => panic!("ConvNode: missing input(s) - x={} w={}", self.x, self.w),
        }

        if let Some(next) = &self.next_node {
            next.pass(omap);
        }
    }

    fn print(&self) {
        println!("conv-{},{},{:?},{}", self.x, self.w, self.b, self.o);

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
