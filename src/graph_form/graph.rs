use std::collections::HashSet;

use crate::graph_form::nodes::{
    add::AddNode, concat::ConcatNode, conv::ConvNode, div::DivNode, max_pool::MaxPoolNode,
    mul::MulNode, node::Node, reshape::ReshapeNode, resize::ResizeNode, sigmoid::SigmoidNode,
    slice::SliceNode, soft_max::SoftMaxNode, split::SplitNode, sub::SubNode,
    transpose::TransposeNode,
};
use anyhow::{Ok, Result};
use onnx_extractor::OnnxModel;

pub struct GraphForm {
    start_node: Option<Box<dyn Node>>,
}

impl GraphForm {
    pub fn new() -> Self {
        Self { start_node: None }
    }

    pub fn insert(&mut self, next: Box<dyn Node>) -> Result<()> {
        if let Some(start_node) = &mut self.start_node {
            start_node.insert(next)?;
        } else {
            self.start_node = Some(next)
        }
        Ok(())
    }

    pub fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.start_node {
            next.self_count(count + 1)
        } else {
            count
        }
    }

    pub fn from_onnx_file(onnx_file_path: &str) -> Result<Self> {
        let onnx = OnnxModel::load_from_file(onnx_file_path)?;

        println!("{}",onnx.execution_order().unwrap().len());

        let mut ret = Self::new();

        onnx.execution_order()?.into_iter().for_each(|elem| {
            println!("{},{:?}",elem.op_type,elem.attributes);
            match elem.op_type.as_str() {
                "Concat" => {
                    ret.insert(Box::new(ConcatNode::default())).unwrap();
                }
                "Sigmoid" => {
                    ret.insert(Box::new(SigmoidNode::default())).unwrap();
                }
                "Conv" => {
                    ret.insert(Box::new(ConvNode::default())).unwrap();
                }
                "Resize" => {
                    ret.insert(Box::new(ResizeNode::default())).unwrap();
                }
                "Transpose" => {
                    ret.insert(Box::new(TransposeNode::default())).unwrap();
                }
                "Sub" => {
                    ret.insert(Box::new(SubNode::default())).unwrap();
                }
                "MaxPool" => {
                    ret.insert(Box::new(MaxPoolNode::default())).unwrap();
                }
                "Div" => {
                    ret.insert(Box::new(DivNode::default())).unwrap();
                }
                "Softmax" => {
                    ret.insert(Box::new(SoftMaxNode::default())).unwrap();
                }
                "Split" => {
                    ret.insert(Box::new(SplitNode::default())).unwrap();
                }
                "Add" => {
                    ret.insert(Box::new(AddNode::default())).unwrap();
                }
                "Mul" => {
                    ret.insert(Box::new(MulNode::default())).unwrap();
                }
                "Reshape" => {
                    ret.insert(Box::new(ReshapeNode::default())).unwrap();
                }
                "Slice" => {
                    ret.insert(Box::new(SliceNode::default())).unwrap();
                }
                _ => {}
            }
        });

        // let mut hash = HashSet::new();
        // onnx.execution_order()?.into_iter().for_each(|element| {
        //     hash.insert(element.op_type.clone());
        // });

        // hash.iter()
        //     .for_each(|elem| println!("\"{}\" => {{}},", elem));

        Ok(ret)
    }
}
