use crate::graph_form::nodes::{hash_trait::FromHashMap, node::Node};
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
pub struct ConvNode {
    x: String,
    w: String,
    b: Option<String>,

    o: String,

    auto_pad: AutoPad,
    kernel_shape: Vec<i64>,
    group: i64,
    pads: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,

    next_node: Option<Box<dyn Node>>,
}

impl FromHashMap for ConvNode {
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
            group: {
                match attrs.get("groups") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            next_node: None,
        })
    }
}

impl ConvNode {
    pub fn new(
        auto_pad: &str,
        kernel_shape: Vec<i64>,
        group: i64,
        pads: Vec<i64>,
        strides: Vec<i64>,
        dilations: Vec<i64>,
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

impl Node for ConvNode {
    fn pass(&self) {
        todo!()
    }

    fn print(&self) {
        println!(
            "conv-{:?},{:?},{:?},{:?},{:?},{:?}",
            self.auto_pad, self.dilations, self.group, self.kernel_shape, self.pads, self.strides
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
