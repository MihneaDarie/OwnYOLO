use crate::yolo::buffers::*;
use anyhow::Result;
use ndarray::{Array1, Array3, Array4, ArrayD, IxDyn, s};
use ndarray_npy::NpzReader;
use std::fs::File;

pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone)]
pub struct ConvBnWeights {
    pub conv_weight: Array4<f32>,
    pub conv_bias: Option<Array1<f32>>,
    pub bn_weight: Array1<f32>,
    pub bn_bias: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct ConvWeights {
    pub weight: Array4<f32>,
    pub bias: Option<Array1<f32>>,
}

#[derive(Debug, Clone)]
pub struct C2fWeights {
    pub cv1: ConvBnWeights,
    pub cv2: ConvBnWeights,
    pub bottlenecks: Vec<BottleneckWeights>,
}

#[derive(Debug, Clone)]
pub struct BottleneckWeights {
    pub cv1: ConvBnWeights,
    pub cv2: ConvBnWeights,
}

#[derive(Debug, Clone)]
pub struct SPPFWeights {
    pub cv1: ConvBnWeights,
    pub cv2: ConvBnWeights,
}

#[derive(Debug, Clone)]
pub struct DetectWeights {
    pub cv2: Vec<DetectConvBlock>,
    pub cv3: Vec<DetectConvBlock>,
    pub dfl_conv_weight: Array4<f32>,
}

#[derive(Debug, Clone)]
pub struct DetectConvBlock {
    pub conv0: ConvBnWeights,
    pub conv1: ConvBnWeights,
    pub conv2: ConvWeights,
}

pub struct YoloV8Weights {
    pub model_0: ConvBnWeights,
    pub model_1: ConvBnWeights,
    pub model_2: C2fWeights,
    pub model_3: ConvBnWeights,
    pub model_4: C2fWeights,
    pub model_5: ConvBnWeights,
    pub model_6: C2fWeights,
    pub model_7: ConvBnWeights,
    pub model_8: C2fWeights,
    pub model_9: SPPFWeights,
    pub model_12: C2fWeights,
    pub model_15: C2fWeights,
    pub model_16: ConvBnWeights,
    pub model_18: C2fWeights,
    pub model_19: ConvBnWeights,
    pub model_21: C2fWeights,
    pub model_22: DetectWeights,
}

impl YoloV8Weights {
    pub fn from_npz(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)?;

        let load_array4 = |npz: &mut NpzReader<File>, name: &str| -> Result<Array4<f32>> {
            let arr: ArrayD<f32> = npz.by_name(&format!("{}.npy", name))?;
            Ok(arr.into_dimensionality()?)
        };

        let load_array1 = |npz: &mut NpzReader<File>, name: &str| -> Result<Array1<f32>> {
            let arr: ArrayD<f32> = npz.by_name(&format!("{}.npy", name))?;
            Ok(arr.into_dimensionality()?)
        };

        let load_conv_bn = |npz: &mut NpzReader<File>, prefix: &str| -> Result<ConvBnWeights> {
            let conv_weight = load_array4(npz, &format!("{}.conv.weight", prefix))?;
            let out_channels = conv_weight.shape()[0];

            let conv_bias = load_array1(npz, &format!("{}.conv.bias", prefix)).ok();

            let (bn_weight, bn_bias) = match (
                load_array1(npz, &format!("{}.bn.weight", prefix)),
                load_array1(npz, &format!("{}.bn.bias", prefix)),
            ) {
                (Ok(weight), Ok(bias)) => (weight, bias),
                _ => (
                    Array1::from_elem(out_channels, 1.0),
                    Array1::from_elem(out_channels, 0.0),
                ),
            };

            Ok(ConvBnWeights {
                conv_weight,
                conv_bias,
                bn_weight,
                bn_bias,
            })
        };

        let load_bottleneck =
            |npz: &mut NpzReader<File>, prefix: &str| -> Result<BottleneckWeights> {
                Ok(BottleneckWeights {
                    cv1: load_conv_bn(npz, &format!("{}.cv1", prefix))?,
                    cv2: load_conv_bn(npz, &format!("{}.cv2", prefix))?,
                })
            };

        let load_c2f = |npz: &mut NpzReader<File>, prefix: &str, n: usize| -> Result<C2fWeights> {
            let mut bottlenecks = Vec::new();
            for i in 0..n {
                bottlenecks.push(load_bottleneck(npz, &format!("{}.m.{}", prefix, i))?);
            }
            Ok(C2fWeights {
                cv1: load_conv_bn(npz, &format!("{}.cv1", prefix))?,
                cv2: load_conv_bn(npz, &format!("{}.cv2", prefix))?,
                bottlenecks,
            })
        };

        let load_detect_block =
            |npz: &mut NpzReader<File>, prefix: &str| -> Result<DetectConvBlock> {
                Ok(DetectConvBlock {
                    conv0: load_conv_bn(npz, &format!("{}.0", prefix))?,
                    conv1: load_conv_bn(npz, &format!("{}.1", prefix))?,
                    conv2: ConvWeights {
                        weight: load_array4(npz, &format!("{}.2.weight", prefix))?,
                        bias: Some(load_array1(npz, &format!("{}.2.bias", prefix))?),
                    },
                })
            };

        Ok(Self {
            model_0: load_conv_bn(&mut npz, "model.0")?,
            model_1: load_conv_bn(&mut npz, "model.1")?,
            model_2: load_c2f(&mut npz, "model.2", 1)?,
            model_3: load_conv_bn(&mut npz, "model.3")?,
            model_4: load_c2f(&mut npz, "model.4", 2)?,
            model_5: load_conv_bn(&mut npz, "model.5")?,
            model_6: load_c2f(&mut npz, "model.6", 2)?,
            model_7: load_conv_bn(&mut npz, "model.7")?,
            model_8: load_c2f(&mut npz, "model.8", 1)?,
            model_9: SPPFWeights {
                cv1: load_conv_bn(&mut npz, "model.9.cv1")?,
                cv2: load_conv_bn(&mut npz, "model.9.cv2")?,
            },
            model_12: load_c2f(&mut npz, "model.12", 1)?,
            model_15: load_c2f(&mut npz, "model.15", 1)?,
            model_16: load_conv_bn(&mut npz, "model.16")?,
            model_18: load_c2f(&mut npz, "model.18", 1)?,
            model_19: load_conv_bn(&mut npz, "model.19")?,
            model_21: load_c2f(&mut npz, "model.21", 1)?,
            model_22: DetectWeights {
                cv2: vec![
                    load_detect_block(&mut npz, "model.22.cv2.0")?,
                    load_detect_block(&mut npz, "model.22.cv2.1")?,
                    load_detect_block(&mut npz, "model.22.cv2.2")?,
                ],
                cv3: vec![
                    load_detect_block(&mut npz, "model.22.cv3.0")?,
                    load_detect_block(&mut npz, "model.22.cv3.1")?,
                    load_detect_block(&mut npz, "model.22.cv3.2")?,
                ],
                dfl_conv_weight: load_array4(&mut npz, "model.22.dfl.conv.weight")?,
            },
        })
    }
}

pub struct YoloV8 {
    pub weights: YoloV8Weights,
}

impl YoloV8 {
    pub fn new(weights_path: &str) -> Result<Self> {

        let weights = YoloV8Weights::from_npz(weights_path)?;
        Ok(Self {
            weights,
        })
    }

    pub fn forward(&self, x: &Array4<f32>, buffers: &mut Buffers ) -> Result<()> {

        

        Ok(())
    }
}
