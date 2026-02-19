use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
use anyhow::Result;
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
pub struct MaxPoolNode {
    auto_pad: AutoPad,
    ceil_mode: i64,
    kernel_shape: Vec<i64>,
    dilations: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    storage_order: i64,
    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for MaxPoolNode {
    fn from_hashmap(
        attrs: &std::collections::HashMap<String, AttributeValue>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
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
                    Some(av) => av.as_ints().unwrap().to_vec(),
                    None => vec![],
                }
            },
            pads: {
                match attrs.get("pads") {
                    Some(av) => av.as_ints().unwrap().to_vec(),
                    None => vec![],
                }
            },
            strides: {
                match attrs.get("strides") {
                    Some(av) => av.as_ints().unwrap().to_vec(),
                    None => vec![],
                }
            },
            dilations: {
                match attrs.get("dilations") {
                    Some(av) => av.as_ints().unwrap().to_vec(),
                    None => vec![],
                }
            },
            ceil_mode: {
                match attrs.get("ceil_mode") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            storage_order: {
                match attrs.get("storage_order") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            next_node: None,
        })
    }
}

impl MaxPoolNode {
    pub fn new(
        auto_pad: &str,
        ceil_mode: i64,
        kernel_shape: Vec<i64>,
        dilations: Vec<i64>,
        strides: Vec<i64>,
        storage_order: i64,
        pads: Vec<i64>,
    ) -> Self {
        Self {
            auto_pad: AutoPad::from_str(auto_pad),
            ceil_mode,
            kernel_shape,
            dilations,
            strides,
            pads,
            storage_order,
            next_node: None,
        }
    }
}

impl Node for MaxPoolNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!(
            "max_pool-{:?},{},{:?},{:?},{:?},{},{:?}",
            self.auto_pad,
            self.ceil_mode,
            self.dilations,
            self.kernel_shape,
            self.pads,
            self.storage_order,
            self.strides
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
    fn insert(&mut self, next: Box<dyn Node>) -> Result<()> {
        if let Some(next_node) = &mut self.next_node {
            next_node.insert(next)?;
            return Ok(());
        } else {
            self.next_node = Some(next)
        }
        Ok(())
    }
}
