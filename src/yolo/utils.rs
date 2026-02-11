use anyhow::Result;
use gemm::Parallelism;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array4};
use rayon::prelude::*;

use crate::yolo::gemms::gemm::sgemm_bias_parallel;

use super::buffers::C2FBuffer;
use super::yolov8::C2fWeights;

pub static FFI: bool = false;

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
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
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    dst.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(d, s)| *d += *s);
}

#[derive(Clone, Copy)]
pub struct Conv2D {
    pub pad: usize,
    pub stride: usize,
}

pub const CFG_1X1_S1_P0: Conv2D = Conv2D { pad: 0, stride: 1 };
pub const CFG_3X3_S1_P1: Conv2D = Conv2D { pad: 1, stride: 1 };

#[inline]
fn im2col_3x3_s1p1(input: &[f32], h: usize, w: usize, col_buffer: &mut [f32]) {
    let hw = h * w;
    col_buffer
        .par_chunks_mut(9 * hw)
        .enumerate()
        .for_each(|(ic, chunk)| {
            let in_c_base = ic * hw;

            for ky in 0..3usize {
                for kx in 0..3usize {
                    let k_idx = ky * 3 + kx;
                    let col_row = &mut chunk[k_idx * hw..(k_idx + 1) * hw];
                    let dy = ky as isize - 1;
                    let dx = kx as isize - 1;

                    for oy in 0..h {
                        let iy = oy as isize + dy;
                        let out_row_start = oy * w;

                        if iy < 0 || iy >= h as isize {
                            for ox in 0..w {
                                col_row[out_row_start + ox] = 0.0;
                            }
                        } else {
                            let in_row_base = in_c_base + (iy as usize) * w;
                            for ox in 0..w {
                                let ix = ox as isize + dx;
                                col_row[out_row_start + ox] = if ix < 0 || ix >= w as isize {
                                    0.0
                                } else {
                                    unsafe { *input.get_unchecked(in_row_base + ix as usize) }
                                };
                            }
                        }
                    }
                }
            }
        });
}

#[inline]
fn im2col_3x3_s2p1(
    input: &[f32],
    hin: usize,
    win: usize,
    hout: usize,
    wout: usize,
    col_buffer: &mut [f32],
) {
    let hw_out = hout * wout;

    col_buffer
        .par_chunks_mut(9 * hw_out)
        .enumerate()
        .for_each(|(ic, chunk)| {
            let in_c_base = ic * hin * win;

            for ky in 0..3usize {
                for kx in 0..3usize {
                    let k_idx = ky * 3 + kx;
                    let col_row = &mut chunk[k_idx * hw_out..(k_idx + 1) * hw_out];

                    for oy in 0..hout {
                        let iy = (oy * 2 + ky) as isize - 1;
                        let out_row_start = oy * wout;

                        if iy < 0 || iy >= hin as isize {
                            for ox in 0..wout {
                                col_row[out_row_start + ox] = 0.0;
                            }
                        } else {
                            let in_row_base = in_c_base + (iy as usize) * win;
                            for ox in 0..wout {
                                let ix = (ox * 2 + kx) as isize - 1;
                                col_row[out_row_start + ox] = if ix < 0 || ix >= win as isize {
                                    0.0
                                } else {
                                    unsafe { *input.get_unchecked(in_row_base + ix as usize) }
                                };
                            }
                        }
                    }
                }
            }
        });
}

thread_local! {
    static IM2COL_BUF_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {std::cell::RefCell::new(Vec::new())};
}

#[inline(always)]
fn run_func_with_f32_buffer<R>(buf_size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    IM2COL_BUF_POOL.with(|cell| {
        let mut buf = cell.borrow_mut().pop().unwrap_or_default();

        if buf.len() < buf_size {
            buf.resize(buf_size, 0.0f32);
        }

        let a = f(&mut buf[..buf_size]);

        cell.borrow_mut().push(buf);

        a
    })
}

pub fn conv_silu_into(
    x: &Array4<f32>,
    w: &Array4<f32>,
    conv_bias: Option<&Array1<f32>>,
    cfg: &Conv2D,
    out: &mut Array4<f32>,
) -> Result<()> {
    let (_, cin, hin, win) = x.dim();
    let (cout, _, kh, kw) = w.dim();

    if kh == 1 && kw == 1 && cfg.pad == 0 && cfg.stride == 1 {
        let hw = hin * win;
        let xs = x.as_slice_memory_order().unwrap();
        let ws = w.as_slice_memory_order().unwrap();
        let out_sl = out.as_slice_memory_order_mut().unwrap();
        let bias = conv_bias.map(|b| b.as_slice().unwrap());

        sgemm_bias_parallel(cout, hw, cin, ws, xs, bias, out_sl, true);
        return Ok(());
    }

    let hout = (hin + 2 * cfg.pad - kh) / cfg.stride + 1;
    let wout = (win + 2 * cfg.pad - kw) / cfg.stride + 1;
    let hw_out = hout * wout;

    let xs = x.as_slice_memory_order().unwrap();
    let ws = w.as_slice_memory_order().unwrap();
    let out_sl = out.as_slice_memory_order_mut().unwrap();

    let col_size = cin * 9 * hw_out;
    run_func_with_f32_buffer(col_size, |col_buffer| {
        if cfg.stride == 1 && cfg.pad == 1 {
            im2col_3x3_s1p1(xs, hin, win, col_buffer);
        } else if cfg.stride == 2 && cfg.pad == 1 {
            im2col_3x3_s2p1(xs, hin, win, hout, wout, col_buffer);
        }

        let k_dim = cin * 9;
        let bias = conv_bias.map(|b| b.as_slice().unwrap());

        if FFI {
            let m = cout;
            let n = hw_out;
            let k = cin * 9;

            if let Some(b) = conv_bias {
                out_sl
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(oc, row)| row.fill(b[oc]));
            } else {
                out_sl.fill(0.0);
            }

            unsafe {
                gemm::gemm::<f32>(
                    m,
                    n,
                    k,
                    out_sl.as_mut_ptr(),
                    1,
                    n as isize,
                    true,
                    ws.as_ptr(),
                    1,
                    k as isize,
                    col_buffer.as_ptr(),
                    1,
                    n as isize,
                    1.0,
                    1.0,
                    false,
                    false,
                    false,
                    Parallelism::Rayon(0),
                );
            }
            out_sl.par_iter_mut().for_each(|v| *v = silu(*v));
        } else {
            sgemm_bias_parallel(cout, hw_out, k_dim, ws, col_buffer, bias, out_sl, true);
        }
    });

    Ok(())
}

pub fn conv1x1_silu_into_blocks(
    w: &Array4<f32>,
    bias: Option<&Array1<f32>>,
    blocks: &[(&[f32], usize)],
    out: &mut Array4<f32>,
) -> Result<()> {
    let (cout, cin, _, _) = w.dim();
    let (_, _, h, wout) = out.dim();
    let hw = h * wout;

    let wsl = w.as_slice_memory_order().unwrap();
    let out_sl = out.as_slice_memory_order_mut().unwrap();

    let total_channels: usize = blocks.iter().map(|(_, k)| *k).sum();

    run_func_with_f32_buffer(total_channels * hw, |concat_input| {
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

        sgemm_bias_parallel(cout, hw, cin, wsl, concat_input, bias_slice, out_sl, true);
    });

    Ok(())
}

pub fn c2f_into(
    x: &Array4<f32>,
    w: &C2fWeights,
    buf: &mut C2FBuffer,
    shortcut: bool,
) -> Result<()> {
    conv_silu_into(
        x,
        &w.cv1.conv_weight,
        w.cv1.conv_bias.as_ref(),
        &CFG_1X1_S1_P0,
        &mut buf.initial,
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

        conv_silu_into(
            &buf.split_1,
            &bw.cv1.conv_weight,
            bw.cv1.conv_bias.as_ref(),
            &CFG_3X3_S1_P1,
            &mut bbuf.cv1_out,
        )?;

        conv_silu_into(
            &bbuf.cv1_out,
            &bw.cv2.conv_weight,
            bw.cv2.conv_bias.as_ref(),
            &CFG_3X3_S1_P1,
            &mut bbuf.cv2_out,
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
    )?;

    Ok(())
}
