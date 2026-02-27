use crate::graph_form::{
    nodes::{
        add::AddNode, concat::ConcatNode, conv::ConvNode, div::DivNode, hash_trait::FromHashMap,
        max_pool::MaxPoolNode, mul::MulNode, node::Node, reshape::ReshapeNode, resize::ResizeNode,
        sigmoid::SigmoidNode, slice::SliceNode, soft_max::SoftMaxNode, split::SplitNode,
        sub::SubNode, transpose::TransposeNode,
    },
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Ok;
use ndarray::ArrayD;
use onnx_extractor::OnnxModel;

pub struct GraphForm<T: Default> {
    // nodes: Vec<Box<dyn Node<T>>>,
    nodes: Option<Box<dyn Node<T>>>,
}

impl<T: Default + 'static> GraphForm<T> {
    pub fn new() -> Self {
        Self { nodes: None }
    }

    pub fn insert(&mut self, node: Box<dyn Node<T>>) {
        if let Some(next) = &mut self.nodes {
            next.insert(node).unwrap();
        } else {
            self.nodes = Some(node)
        }
    }

    pub fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.nodes {
            next.self_count(count + 1)
        } else {
            count
        }
    }

    pub fn print(&self) {
        println!("start!");
        if let Some(next) = &self.nodes {
            next.print();
        }
    }

    pub fn load_data_arrays(onnx: &OnnxModel) -> TensorMap {
        let mut map = TensorMap::new();

        onnx.tensor_names().iter().for_each(|t| {
            if let Some(tensor) = onnx.get_tensor(t) {
                let typed = if tensor.data().is_ok() {
                    TypedArray::from_tensor(&tensor)
                } else {
                    TypedArray::from_tensor_empty(tensor)
                };
                map.insert(tensor.name().to_string(), typed);
            }
        });

        map
    }

    pub fn from_onnx_file(onnx_file_path: &str) -> anyhow::Result<(Self, TensorMap)> {
        let onnx = OnnxModel::load_from_file(onnx_file_path)?;

        let mut ret = Self::new();
        let map = Self::load_data_arrays(&onnx);

        onnx.execution_order()?
            .into_iter()
            .for_each(|elem| match elem.op_type.as_str() {
                "Concat" => {
                    let mut concat = ConcatNode::from_hashmap(&elem.attributes).unwrap();
                    concat.add_input_strings(elem.inputs.clone());
                    concat.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(concat));
                }
                "Sigmoid" => {
                    let mut sigmoid = SigmoidNode::new();
                    sigmoid.add_input_strings(elem.inputs[0].clone());
                    sigmoid.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sigmoid));
                }
                "Conv" => {
                    let mut conv = ConvNode::from_hashmap(&elem.attributes).unwrap();
                    let inputs = &elem.inputs;
                    let b = inputs.get(2).map(|s| s.clone());
                    conv.add_input_strings(inputs[0].clone(), inputs[1].clone(), b);
                    conv.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(conv));
                }
                "Resize" => {
                    let inputs = &elem.inputs;
                    let roi = inputs.get(1).filter(|s| !s.is_empty()).cloned();
                    let scales = inputs.get(2).filter(|s| !s.is_empty()).cloned();
                    let sizes = inputs.get(3).filter(|s| !s.is_empty()).cloned();

                    let mut resize = ResizeNode::from_hashmap(&elem.attributes).unwrap();
                    resize.add_input_strings(inputs[0].clone(), roi, scales, sizes);
                    resize.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(resize));
                }
                "Transpose" => {
                    let mut trans = TransposeNode::from_hashmap(&elem.attributes).unwrap();
                    trans.add_input_strings(elem.inputs[0].clone());
                    trans.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(trans));
                }
                "Sub" => {
                    let mut sub = SubNode::new();
                    sub.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    sub.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sub));
                }
                "MaxPool" => {
                    let mut max_pool = MaxPoolNode::from_hashmap(&elem.attributes).unwrap();
                    max_pool.add_input_strings(elem.inputs[0].clone());
                    max_pool.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(max_pool));
                }
                "Div" => {
                    let mut div = DivNode::new();
                    div.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    div.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(div));
                }
                "Softmax" => {
                    let mut soft_max = SoftMaxNode::from_hashmap(&elem.attributes).unwrap();
                    soft_max.add_input_strings(elem.inputs[0].clone());
                    soft_max.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(soft_max));
                }
                "Split" => {
                    let mut split = SplitNode::from_hashmap(&elem.attributes).unwrap();
                    split.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    split.add_output_strings(elem.outputs.clone());
                    ret.insert(Box::new(split));
                }
                "Add" => {
                    let mut add = AddNode::new();
                    add.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    add.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(add));
                }
                "Mul" => {
                    let mut mul = MulNode::new();
                    mul.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    mul.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(mul));
                }
                "Reshape" => {
                    let mut reshape = ReshapeNode::from_hashmap(&elem.attributes).unwrap();
                    reshape.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    reshape.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(reshape));
                }
                "Slice" => {
                    let mut slice = SliceNode::new();
                    let input = &elem.inputs;
                    slice.add_input_strings(
                        input[0].clone(),
                        input[1].clone(),
                        input[2].clone(),
                        input[3].clone(),
                    );
                    slice.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(slice));
                }
                _ => {}
            });

        Ok((ret, map))
    }

    pub fn optimize_nodes(&mut self) {
        if let Some(node) = &mut self.nodes {
            node.optimize_further();
        }
    }

    pub fn pass(&self, omap: &mut TensorMap, input: &ArrayD<f32>) {
        omap.insert("images".to_string(), TypedArray::F32(input.clone()));

        if let Some(next_node) = &self.nodes {
            next_node.pass(omap);
        }
    }
}
