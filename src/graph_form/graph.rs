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
            println!("{},{}", elem.op_type, elem.inputs.len());
            elem.inputs.iter().for_each(|val| println!("{}", val));
            match elem.op_type.as_str() {
                "Concat" => {
                    let mut concat = ConcatNode::from_hashmap(&elem.attributes).unwrap();
                    concat.add_input_strings(elem.inputs.clone());
                    concat.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(concat)).unwrap();
                }
                "Sigmoid" => {
                    let mut sigmoid = SigmoidNode::default();
                    sigmoid.add_input_strings(elem.inputs[0].clone());
                    sigmoid.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sigmoid)).unwrap();
                }
                "Conv" => {
                    let mut conv = ConvNode::from_hashmap(&elem.attributes).unwrap();
                    let inputs = &elem.inputs;
                    let b = inputs.get(2).map(|s| s.clone());
                    conv.add_input_strings(inputs[0].clone(), inputs[1].clone(), b);
                    conv.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(conv)).unwrap();
                }
                "Resize" => {
                    let mut resize = ResizeNode::from_hashmap(&elem.attributes).unwrap();
                    resize.add_input_strings(elem.inputs[0].clone(), None, None, None);
                    resize.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(resize)).unwrap();
                }
                "Transpose" => {
                    let mut trans = TransposeNode::from_hashmap(&elem.attributes).unwrap();
                    trans.add_input_strings(elem.inputs[0].clone());
                    trans.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(trans)).unwrap();
                }
                "Sub" => {
                    let mut sub = SubNode::default();
                    sub.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    sub.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sub)).unwrap();
                }
                "MaxPool" => {
                    let mut max_pool = MaxPoolNode::from_hashmap(&elem.attributes).unwrap();
                    max_pool.add_input_strings(elem.inputs[0].clone());
                    max_pool.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(max_pool)).unwrap();
                }
                "Div" => {
                    let mut div = DivNode::default();
                    div.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    div.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(div)).unwrap();
                }
                "Softmax" => {
                    let mut soft_max = SoftMaxNode::from_hashmap(&elem.attributes).unwrap();
                    soft_max.add_input_strings(elem.inputs[0].clone());
                    soft_max.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(soft_max)).unwrap();
                }
                "Split" => {
                    let mut split = SplitNode::from_hashmap(&elem.attributes).unwrap();

                    split.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    split.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(split)).unwrap();
                }
                "Add" => {
                    let mut add = AddNode::default();
                    add.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    add.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(add)).unwrap();
                }
                "Mul" => {
                    let mut mul = MulNode::default();
                    mul.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    mul.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(mul)).unwrap();
                }
                "Reshape" => {
                    let mut reshape = ReshapeNode::from_hashmap(&elem.attributes).unwrap();
                    reshape.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    reshape.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(reshape)).unwrap();
                }
                "Slice" => {
                    let mut slice = SliceNode::default();
                    let input = &elem.inputs;
                    slice.add_input_strings(
                        input[0].clone(),
                        input[1].clone(),
                        input[2].clone(),
                        input[3].clone(),
                    );
                    slice.add_output_strings(elem.outputs[0].clone());
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
