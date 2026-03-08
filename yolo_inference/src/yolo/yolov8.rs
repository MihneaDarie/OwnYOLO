use crate::yolo::{
    buffers::*,
    utils::{c2f_into, sigmoid},
};
use anyhow::Result;
use ndarray::{Array1, Array3, Array4, ArrayD, ArrayView4, ArrayViewMut4};
use ndarray_npy::NpzReader;
use onnx_graph::{nodes::conv::Conv2D, typed_array::{TypedArray, maxpool_5x5}};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    join,
    slice::ParallelSliceMut,
};
use saker_rs::{activations::Activation, linarg::operations::sgemm_bias_parallel};
use std::{cell::RefCell, fs::File};

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

            let conv_bias = load_array1(npz, &format!("{}.conv.bias", prefix)).ok();

            Ok(ConvBnWeights {
                conv_weight,
                conv_bias,
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
            },
        })
    }
}

pub const CONV_3X3_S2: Conv2D = Conv2D { pad: 1, stride: 2 };
pub const CONV_3X3_S1: Conv2D = Conv2D { pad: 1, stride: 1 };
pub const CONV_1X1_S1: Conv2D = Conv2D { pad: 0, stride: 1 };

#[inline(always)]
fn max5(a: &f32, b: &f32, c: &f32, d: &f32, e: &f32) -> f32 {
    a.max(*b).max(*c).max(*d).max(*e)
}

pub struct YoloV8 {
    pub weights: YoloV8Weights,
    anchors: Vec<(f32, f32)>,
    anchor_stride: Vec<f32>,
}

impl YoloV8 {
    pub fn new(weights_path: &str) -> Result<Self> {
        let weights = YoloV8Weights::from_npz(weights_path)?;

        let mut anchors = Vec::with_capacity(8400);
        let mut anchor_stride = Vec::with_capacity(8400);

        for (stride, grid_size) in [(8.0f32, 80usize), (16.0, 40), (32.0, 20)] {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    anchors.push(((x as f32 + 0.5) * stride, (y as f32 + 0.5) * stride));
                    anchor_stride.push(stride);
                }
            }
        }

        Ok(Self {
            weights,
            anchors,
            anchor_stride,
        })
    }

    pub fn forward(&self, x: &Array4<f32>, buffers: &mut Buffers) -> Result<()> {
        // BACKBONE
        TypedArray::conv_silu_into(
            &x.view(),
            &self.weights.model_0.conv_weight.view(),
            self.weights.model_0.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_0_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;
        TypedArray::conv_silu_into(
            &buffers.model_0_buffer.conv_out.view(),
            &self.weights.model_1.conv_weight.view(),
            self.weights.model_1.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_1_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;

        c2f_into(
            &buffers.model_1_buffer.conv_out,
            &self.weights.model_2,
            &mut buffers.model_2_buffer,
            true,
            Activation::Silu,
        )?;

        TypedArray::conv_silu_into(
            &buffers.model_2_buffer.last.view(),
            &self.weights.model_3.conv_weight.view(),
            self.weights.model_3.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_3_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;

        c2f_into(
            &buffers.model_3_buffer.conv_out,
            &self.weights.model_4,
            &mut buffers.model_4_buffer,
            true,
            Activation::Silu,
        )?;

        TypedArray::conv_silu_into(
            &buffers.model_4_buffer.last.view(),
            &self.weights.model_5.conv_weight.view(),
            self.weights.model_5.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_5_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;

        c2f_into(
            &buffers.model_5_buffer.conv_out,
            &self.weights.model_6,
            &mut buffers.model_6_buffer,
            true,
            Activation::Silu,
        )?;

        TypedArray::conv_silu_into(
            &buffers.model_6_buffer.last.view(),
            &self.weights.model_7.conv_weight.view(),
            self.weights.model_7.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_7_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;

        c2f_into(
            &buffers.model_7_buffer.conv_out,
            &self.weights.model_8,
            &mut buffers.model_8_buffer,
            true,
            Activation::Silu,
        )?;

        self.sppf_into(
            &buffers.model_8_buffer.last,
            &self.weights.model_9,
            &mut buffers.model_9_buffer,
        )?;

        // NECK
        upsample_2x(
            &buffers.model_9_buffer.cv2_out,
            &mut buffers.up1_buffer.output,
        );

        concat_channels(
            &buffers.up1_buffer.output,
            &buffers.model_6_buffer.last,
            &mut buffers.concat1_buffer.output,
        );

        c2f_into(
            &buffers.concat1_buffer.output,
            &self.weights.model_12,
            &mut buffers.model_12_buffer,
            false,
            Activation::Silu,
        )?;

        upsample_2x(
            &buffers.model_12_buffer.last,
            &mut buffers.up2_buffer.output,
        );

        concat_channels(
            &buffers.up2_buffer.output,
            &buffers.model_4_buffer.last,
            &mut buffers.concat2_buffer.output,
        );

        c2f_into(
            &buffers.concat2_buffer.output,
            &self.weights.model_15,
            &mut buffers.model_15_buffer,
            false,
            Activation::Silu,
        )?;

        TypedArray::conv_silu_into(
            &buffers.model_15_buffer.last.view(),
            &self.weights.model_16.conv_weight.view(),
            self.weights.model_16.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_16_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;
        concat_channels(
            &buffers.model_16_buffer.conv_out,
            &buffers.model_12_buffer.last,
            &mut buffers.concat3_buffer.output,
        );
        c2f_into(
            &buffers.concat3_buffer.output,
            &self.weights.model_18,
            &mut buffers.model_18_buffer,
            false,
            Activation::Silu,
        )?;

        TypedArray::conv_silu_into(
            &buffers.model_18_buffer.last.view(),
            &self.weights.model_19.conv_weight.view(),
            self.weights.model_19.conv_bias.as_ref().map(|a| a.view()),
            &CONV_3X3_S2,
            &mut buffers.model_19_buffer.conv_out.view_mut(),
            Activation::Silu,
        )?;

        concat_channels(
            &buffers.model_19_buffer.conv_out,
            &buffers.model_9_buffer.cv2_out,
            &mut buffers.concat4_buffer.output,
        );

        c2f_into(
            &buffers.concat4_buffer.output,
            &self.weights.model_21,
            &mut buffers.model_21_buffer,
            false,
            Activation::Silu,
        )?;
        let input0 = &buffers.model_15_buffer.last;
        let input1 = &buffers.model_18_buffer.last;
        let input2 = &buffers.model_21_buffer.last;

        let (sb0, sb1, sb2) = {
            let (s0, rest) = buffers
                .model_22_buffer
                .scale_outputs
                .as_mut_slice()
                .split_at_mut(1);
            let (s1, s2) = rest.split_at_mut(1);
            (&mut s0[0], &mut s1[0], &mut s2[0])
        };

        let ((r0, r1), r2) = join(
            || {
                join(
                    || {
                        self.detect_scale_compute_only(
                            input0,
                            &self.weights.model_22.cv2[0],
                            &self.weights.model_22.cv3[0],
                            sb0,
                        )
                    },
                    || {
                        self.detect_scale_compute_only(
                            input1,
                            &self.weights.model_22.cv2[1],
                            &self.weights.model_22.cv3[1],
                            sb1,
                        )
                    },
                )
            },
            || {
                self.detect_scale_compute_only(
                    input2,
                    &self.weights.model_22.cv2[2],
                    &self.weights.model_22.cv3[2],
                    sb2,
                )
            },
        );

        r0?;
        r1?;
        r2?;

        let fc_row = 8400usize;
        let fc = buffers
            .model_22_buffer
            .final_concat
            .as_slice_memory_order_mut()
            .unwrap();

        {
            let (_, _, h, w) = sb0.bbox_out.dim();
            let hw = h * w;
            let bbox_flat = sb0.bbox_out.as_slice_memory_order().unwrap();
            let class_flat = sb0.class_out.as_slice_memory_order().unwrap();
            Self::pack_scale_into_final_concat(bbox_flat, class_flat, hw, fc, fc_row, 0);
        }

        {
            let (_, _, h, w) = sb1.bbox_out.dim();
            let hw = h * w;
            let bbox_flat = sb1.bbox_out.as_slice_memory_order().unwrap();
            let class_flat = sb1.class_out.as_slice_memory_order().unwrap();
            Self::pack_scale_into_final_concat(bbox_flat, class_flat, hw, fc, fc_row, 6400);
        }

        {
            let (_, _, h, w) = sb2.bbox_out.dim();
            let hw = h * w;
            let bbox_flat = sb2.bbox_out.as_slice_memory_order().unwrap();
            let class_flat = sb2.class_out.as_slice_memory_order().unwrap();
            Self::pack_scale_into_final_concat(bbox_flat, class_flat, hw, fc, fc_row, 8000);
        }

        self.decode_predictions(buffers)?;
        Ok(())
    }

    #[inline(always)]
    fn detect_scale_compute_only(
        &self,
        input: &Array4<f32>,
        cv2_block: &DetectConvBlock,
        cv3_block: &DetectConvBlock,
        scale_buf: &mut DetectScaleBuffer,
    ) -> anyhow::Result<()> {
        let DetectScaleBuffer {
            cv2_0_out,
            cv2_1_out,
            bbox_out,
            cv3_0_out,
            cv3_1_out,
            class_out,
            ..
        } = &mut *scale_buf;

        let (r_cv2, r_cv3) = rayon::join(
            || -> anyhow::Result<()> {
                TypedArray::conv_silu_into(
                    &input.view(),
                    &cv2_block.conv0.conv_weight.view(),
                    cv2_block.conv0.conv_bias.as_ref().map(|a| a.view()),
                    &CONV_3X3_S1,
                    &mut cv2_0_out.view_mut(),
                    Activation::Silu,
                )?;

                TypedArray::conv_silu_into(
                    &cv2_0_out.view(),
                    &cv2_block.conv1.conv_weight.view(),
                    cv2_block.conv1.conv_bias.as_ref().map(|a| a.view()),
                    &CONV_3X3_S1,
                    &mut cv2_1_out.view_mut(),
                    Activation::Silu,
                )?;

                conv_linear_into(
                    cv2_1_out,
                    &cv2_block.conv2.weight,
                    cv2_block.conv2.bias.as_ref(),
                    bbox_out,
                    Activation::Silu,
                )?;
                Ok(())
            },
            || -> anyhow::Result<()> {
                TypedArray::conv_silu_into(
                    &input.view(),
                    &cv3_block.conv0.conv_weight.view(),
                    cv3_block.conv0.conv_bias.as_ref().map(|a| a.view()),
                    &CONV_3X3_S1,
                    &mut cv3_0_out.view_mut(),
                    Activation::Silu,
                )?;

                TypedArray::conv_silu_into(
                    &cv3_0_out.view(),
                    &cv3_block.conv1.conv_weight.view(),
                    cv3_block.conv1.conv_bias.as_ref().map(|a| a.view()),
                    &CONV_3X3_S1,
                    &mut cv3_1_out.view_mut(),
                    Activation::Silu,
                )?;

                conv_linear_into(
                    cv3_1_out,
                    &cv3_block.conv2.weight,
                    cv3_block.conv2.bias.as_ref(),
                    class_out,
                    Activation::Silu,
                )?;
                Ok(())
            },
        );

        r_cv2?;
        r_cv3?;
        Ok(())
    }

    #[inline(always)]
    fn pack_scale_into_final_concat(
        bbox_flat: &[f32],
        class_flat: &[f32],
        hw: usize,
        fc: &mut [f32],
        fc_row: usize,
        offset: usize,
    ) {
        for c in 0..64 {
            let src = &bbox_flat[c * hw..(c + 1) * hw];
            let base = c * fc_row + offset;
            fc[base..base + hw].copy_from_slice(src);
        }
        for c in 0..80 {
            let src = &class_flat[c * hw..(c + 1) * hw];
            let base = (64 + c) * fc_row + offset;
            fc[base..base + hw].copy_from_slice(src);
        }
    }

    fn sppf_into(
        &self,
        x: &Array4<f32>,
        weights: &SPPFWeights,
        buf: &mut SPPFBuffer,
    ) -> Result<()> {
        TypedArray::conv_silu_into(
            &x.view(),
            &weights.cv1.conv_weight.view(),
            weights.cv1.conv_bias.as_ref().map(|a| a.view()),
            &CONV_1X1_S1,
            &mut buf.cv1_out.view_mut(),
            Activation::Silu,
        )?;
        maxpool_5x5(&buf.cv1_out.view(), &mut buf.pool_1.view_mut());
        maxpool_5x5(&buf.pool_1.view(), &mut buf.pool_2.view_mut());
        maxpool_5x5(&buf.pool_2.view(), &mut buf.pool_3.view_mut());

        concat_4_channels(
            &buf.cv1_out,
            &buf.pool_1,
            &buf.pool_2,
            &buf.pool_3,
            &mut buf.concat,
        );

        TypedArray::conv_silu_into(
            &buf.concat.view(),
            &weights.cv2.conv_weight.view(),
            weights.cv2.conv_bias.as_ref().map(|a| a.view()),
            &CONV_1X1_S1,
            &mut buf.cv2_out.view_mut(),
            Activation::Silu,
        )?;

        Ok(())
    }

    fn decode_predictions(&self, buffers: &mut Buffers) -> Result<()> {
        let final_concat = &buffers.model_22_buffer.final_concat;
        let (_, _ch, num_anchors) = final_concat.dim();
        let fc = final_concat.as_slice_memory_order().unwrap();

        let bbox_src = &fc[..64 * num_anchors];
        let class_src = &fc[64 * num_anchors..144 * num_anchors];

        let class_pred = &mut buffers.model_22_buffer.class_pred;
        let class_out = class_pred.as_slice_memory_order_mut().unwrap();
        class_out
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dst)| *dst = sigmoid(unsafe { *class_src.get_unchecked(i) }));

        let final_output = &mut buffers.model_22_buffer.final_output;
        let out = final_output.as_slice_memory_order_mut().unwrap();
        let out_ptr_usize = out.as_mut_ptr() as usize;

        let anchors = &self.anchors;
        let strides = &self.anchor_stride;

        (0..num_anchors).into_par_iter().for_each(|a| {
            let (cx, cy) = unsafe { *anchors.get_unchecked(a) };
            let stride = unsafe { *strides.get_unchecked(a) };

            let mut d = [0f32; 4];
            for (coord, value) in d.iter_mut().enumerate() {
                let base = (coord * 16) * num_anchors + a;

                let mut maxv = f32::NEG_INFINITY;
                for bin in 0..16 {
                    let v = unsafe { *bbox_src.get_unchecked(base + bin * num_anchors) };
                    if v > maxv {
                        maxv = v;
                    }
                }

                let mut sum = 0.0f32;
                let mut wsum = 0.0f32;
                for bin in 0..16 {
                    let v = unsafe { *bbox_src.get_unchecked(base + bin * num_anchors) };
                    let e = (v - maxv).exp();
                    sum += e;
                    wsum += e * (bin as f32);
                }

                *value = wsum / sum;
            }

            let left = d[0] * stride;
            let top = d[1] * stride;
            let right = d[2] * stride;
            let bottom = d[3] * stride;

            let x1 = cx - left;
            let y1 = cy - top;
            let x2 = cx + right;
            let y2 = cy + bottom;

            let out_ptr = out_ptr_usize as *mut f32;
            unsafe {
                *out_ptr.add(a) = x1;
                *out_ptr.add(num_anchors + a) = y1;
                *out_ptr.add(2 * num_anchors + a) = x2;
                *out_ptr.add(3 * num_anchors + a) = y2;
            }
        });

        out[4 * num_anchors..84 * num_anchors].copy_from_slice(class_out);

        Ok(())
    }
}

pub fn upsample_2x(input: &Array4<f32>, output: &mut Array4<f32>) {
    let (_, _, h, w) = input.dim();

    let in_sl = input.as_slice_memory_order().unwrap();
    let out_sl = output.as_slice_memory_order_mut().unwrap();

    let in_hw = h * w;
    let out_hw = (h * 2) * (w * 2);

    out_sl
        .par_chunks_mut(out_hw)
        .enumerate()
        .for_each(|(ch, out_ch)| {
            let in_base = ch * in_hw;

            for y in 0..h {
                let in_row = in_base + y * w;
                let out_y = y * 2;

                for x in 0..w {
                    let v = unsafe { *in_sl.get_unchecked(in_row + x) };
                    let out_x = x * 2;

                    let row0 = out_y * (w * 2);
                    let row1 = (out_y + 1) * (w * 2);

                    unsafe {
                        *out_ch.get_unchecked_mut(row0 + out_x) = v;
                        *out_ch.get_unchecked_mut(row0 + out_x + 1) = v;
                        *out_ch.get_unchecked_mut(row1 + out_x) = v;
                        *out_ch.get_unchecked_mut(row1 + out_x + 1) = v;
                    }
                }
            }
        });
}

#[inline(always)]
fn concat_channels(a: &Array4<f32>, b: &Array4<f32>, out: &mut Array4<f32>) {
    let (_, ca, h, w) = a.dim();
    let (_, cb, _, _) = b.dim();
    let hw = h * w;

    let a_sl = a.as_slice_memory_order().unwrap();
    let b_sl = b.as_slice_memory_order().unwrap();
    let o = out.as_slice_memory_order_mut().unwrap();

    o[..ca * hw].copy_from_slice(&a_sl[..ca * hw]);
    o[ca * hw..(ca + cb) * hw].copy_from_slice(&b_sl[..cb * hw]);
}

fn concat_4_channels(
    a: &Array4<f32>,
    b: &Array4<f32>,
    c: &Array4<f32>,
    d: &Array4<f32>,
    output: &mut Array4<f32>,
) {
    let (_, ca, h, w) = a.dim();
    let (_, cb, _, _) = b.dim();
    let (_, cc, _, _) = c.dim();
    let (_, cd, _, _) = d.dim();
    let hw = h * w;

    let out_slice = output.as_slice_memory_order_mut().unwrap();

    let mut offset = 0;
    out_slice[offset..offset + ca * hw]
        .copy_from_slice(&a.as_slice_memory_order().unwrap()[..ca * hw]);
    offset += ca * hw;

    out_slice[offset..offset + cb * hw]
        .copy_from_slice(&b.as_slice_memory_order().unwrap()[..cb * hw]);
    offset += cb * hw;

    out_slice[offset..offset + cc * hw]
        .copy_from_slice(&c.as_slice_memory_order().unwrap()[..cc * hw]);
    offset += cc * hw;

    out_slice[offset..offset + cd * hw]
        .copy_from_slice(&d.as_slice_memory_order().unwrap()[..cd * hw]);
}

pub fn conv_linear_into(
    x: &Array4<f32>,
    w: &Array4<f32>,
    bias: Option<&Array1<f32>>,
    out: &mut Array4<f32>,
    activation: Activation,
) -> Result<()> {
    let (_, cin, h, wout_dim) = x.dim();
    let (cout, _, _, _) = w.dim();
    let hw = h * wout_dim;

    let xs = x.as_slice_memory_order().unwrap();
    let ws = w.as_slice_memory_order().unwrap();
    let out_sl = out.as_slice_memory_order_mut().unwrap();
    let bias_slice = bias.map(|b| b.as_slice().unwrap());

    sgemm_bias_parallel(cout, hw, cin, ws, xs, bias_slice, out_sl, activation);
    Ok(())
}
