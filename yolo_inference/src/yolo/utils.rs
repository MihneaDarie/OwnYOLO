use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array4, ArrayView1, ArrayView4, ArrayViewMut4};
use onnx_graph::nodes::conv::Conv2D;
use onnx_graph::typed_array::TypedArray;
use rayon::prelude::*;
use saker_rs::activations::Activation;
use saker_rs::gemms::operations::sgemm_bias_parallel;


use super::buffers::C2FBuffer;
use super::yolov8::C2fWeights;

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn silu_f32(x: f32) -> f32 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn silu_f64(x: f64) -> f64 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn aprox_sigmoid_f32(x: f32) -> f32 {
    silu_f32(x) / x
}

#[inline(always)]
pub fn aprox_sigmoid_f64(x: f64) -> f64 {
    silu_f64(x) / x
}

#[inline(always)]
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    dst.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(d, s)| *d += *s);
}

pub const CFG_1X1_S1_P0: Conv2D = Conv2D { pad: 0, stride: 1 };
pub const CFG_3X3_S1_P1: Conv2D = Conv2D { pad: 1, stride: 1 };

pub fn c2f_into(
    x: &Array4<f32>,
    w: &C2fWeights,
    buf: &mut C2FBuffer,
    shortcut: bool,
    activation: Activation,
) -> Result<()> {
    TypedArray::conv_silu_into(
        &x.view(),
        &w.cv1.conv_weight.view(),
        w.cv1.conv_bias.as_ref().map(|a| a.view()),
        &CFG_1X1_S1_P0,
        &mut buf.initial.view_mut(),
        activation,
    )?;

    let (_, c2, h, ww) = buf.initial.dim();
    let hidden = c2 / 2;
    let hw = h * ww;

    let init_sl = buf.initial.as_slice_memory_order().unwrap();
    let y0 = &init_sl[0..hidden * hw];
    let y1_init = &init_sl[hidden * hw..2 * hidden * hw];

    buf.split_1
        .as_slice_memory_order_mut()
        .unwrap()
        .copy_from_slice(y1_init);

    let nb = w.bottlenecks.len();
    for i in 0..nb {
        let bbuf = &mut buf.bottlenecks[i];
        let bw = &w.bottlenecks[i];

        TypedArray::conv_silu_into(
            &buf.split_1.view(),
            &bw.cv1.conv_weight.view(),
            bw.cv1.conv_bias.as_ref().map(|array| array.view()),
            &CFG_3X3_S1_P1,
            &mut bbuf.cv1_out.view_mut(),
            activation,
        )?;

        TypedArray::conv_silu_into(
            &bbuf.cv1_out.view(),
            &bw.cv2.conv_weight.view(),
            bw.cv2.conv_bias.as_ref().map(|a| a.view()),
            &CFG_3X3_S1_P1,
            &mut bbuf.cv2_out.view_mut(),
            activation,
        )?;

        if shortcut {
            let src = buf.split_1.as_slice_memory_order().unwrap().to_vec();
            let dst = bbuf.cv2_out.as_slice_memory_order_mut().unwrap();
            add_inplace(dst, &src);
        }

        buf.split_1
            .as_slice_memory_order_mut()
            .unwrap()
            .copy_from_slice(bbuf.cv2_out.as_slice_memory_order().unwrap());
    }

    let empty: &[f32] = &[];
    let b0_out = if nb >= 1 {
        buf.bottlenecks[0].cv2_out.as_slice_memory_order().unwrap()
    } else {
        empty
    };
    let b1_out = if nb >= 2 {
        buf.bottlenecks[1].cv2_out.as_slice_memory_order().unwrap()
    } else {
        empty
    };

    let blocks_all: [(&[f32], usize); 4] = [
        (y0, hidden),
        (y1_init, hidden),
        (b0_out, if nb >= 1 { hidden } else { 0 }),
        (b1_out, if nb >= 2 { hidden } else { 0 }),
    ];

    conv1x1_silu_into_blocks(
        &w.cv2.conv_weight,
        w.cv2.conv_bias.as_ref(),
        &blocks_all,
        &mut buf.last,
        activation,
    )?;

    Ok(())
}

pub fn conv1x1_silu_into_blocks(
    w: &Array4<f32>,
    bias: Option<&Array1<f32>>,
    blocks: &[(&[f32], usize)],
    out: &mut Array4<f32>,
    activation: Activation,
) -> Result<()> {
    let (cout, cin, _, _) = w.dim();
    let (_, _, h, wout) = out.dim();
    let hw = h * wout;

    let wsl = w.as_slice_memory_order().unwrap();
    let out_sl = out.as_slice_memory_order_mut().unwrap();

    let total_channels: usize = blocks.iter().map(|(_, k)| *k).sum();

    TypedArray::run_func_with_f32_buffer(total_channels * hw, |concat_input| {
        let mut offset = 0;
        for (block_data, k) in blocks.iter() {
            if *k > 0 {
                let block_size = *k * hw;
                concat_input[offset..offset + block_size]
                    .copy_from_slice(&block_data[..block_size]);
                offset += block_size;
            }
        }

        let bias_slice = bias.map(|b| b.as_slice().unwrap());

        sgemm_bias_parallel(
            cout,
            hw,
            cin,
            wsl,
            concat_input,
            bias_slice,
            out_sl,
            activation,
        );
    });

    Ok(())
}