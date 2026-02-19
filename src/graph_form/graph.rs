use crate::graph_form::nodes::{
    add::AddNode, concat::ConcatNode, conv::ConvNode, div::DivNode, hash_trait::FromHashMap,
    max_pool::MaxPoolNode, mul::MulNode, node::Node, reshape::ReshapeNode, resize::ResizeNode,
    sigmoid::SigmoidNode, slice::SliceNode, soft_max::SoftMaxNode, split::SplitNode, sub::SubNode,
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

    pub fn print(&self) {
        println!("start!");
        if let Some(next) = &self.start_node {
            next.print();
        }
    }

    pub fn from_onnx_file(onnx_file_path: &str) -> Result<Self> {
        let onnx = OnnxModel::load_from_file(onnx_file_path)?;

        println!("{}", onnx.execution_order().unwrap().len());

        let mut ret = Self::new();

        onnx.execution_order()?.into_iter().for_each(|elem| {
            // println!("{},{:?}", elem.op_type, elem.attributes);
            match elem.op_type.as_str() {
                "Concat" => {
                    let mut concat = ConcatNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(concat)).unwrap();
                }
                "Sigmoid" => {
                    let mut sigmoid = SigmoidNode::default();
                    ret.insert(Box::new(sigmoid)).unwrap();
                }
                "Conv" => {
                    let mut conv = ConvNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(conv)).unwrap();
                }
                "Resize" => {
                    let mut resize = ResizeNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(resize)).unwrap();
                }
                "Transpose" => {
                    let mut trans = TransposeNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(trans)).unwrap();
                }
                "Sub" => {
                    let mut sub = SubNode::default();
                    ret.insert(Box::new(sub)).unwrap();
                }
                "MaxPool" => {
                    let mut max_pool = MaxPoolNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(max_pool)).unwrap();
                }
                "Div" => {
                    let mut div = DivNode::default();
                    ret.insert(Box::new(div)).unwrap();
                }
                "Softmax" => {
                    let mut soft_max = SoftMaxNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(soft_max)).unwrap();
                }
                "Split" => {
                    let mut split = SplitNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(split)).unwrap();
                }
                "Add" => {
                    let mut add = AddNode::default();
                    ret.insert(Box::new(add)).unwrap();
                }
                "Mul" => {
                    let mut mul = MulNode::default();
                    ret.insert(Box::new(mul)).unwrap();
                }
                "Reshape" => {
                    let mut reshape = ReshapeNode::from_hashmap(&elem.attributes).unwrap();
                    ret.insert(Box::new(reshape)).unwrap();
                }
                "Slice" => {
                    let mut slice = SliceNode::default();
                    ret.insert(Box::new(slice)).unwrap();
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
