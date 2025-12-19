use anyhow::{Ok, Result};
use ndarray::{Array1, Array4, ArrayD, IxDyn};

#[derive(Default, Clone, Copy)]
pub struct ConvParams {
    pub padding: u8,
    pub strides: u8,
    pub dilatations: u8,
    pub kernel: u8,
}

pub fn conv2d(
    x: &Array4<f32>,
    w: &Array4<f32>,
    b: Option<&Array1<f32>>,
    params: &ConvParams,
) -> anyhow::Result<ArrayD<f32>> {
    let (n, cin, hin, win) = x.dim();
    let (cout, cin_w, kh, kw) = w.dim();

    anyhow::ensure!(cin == cin_w, "Cin mismatch: x has {}, w has {}", cin, cin_w);
    if let Some(ref bias) = b {
        anyhow::ensure!(bias.len() == cout, "Bias len mismatch");
    }

    let p = params.padding as isize;
    let s = params.strides as isize;
    let d = params.dilatations as isize;

    if params.kernel != 0 {
        anyhow::ensure!(
            params.kernel as usize == kh && params.kernel as usize == kw,
            "Kernel mismatch"
        );
    }

    let eff_kh = d * (kh as isize - 1) + 1;
    let eff_kw = d * (kw as isize - 1) + 1;

    let hout = ((hin as isize + 2 * p - eff_kh) / s + 1) as usize;
    let wout = ((win as isize + 2 * p - eff_kw) / s + 1) as usize;

    let mut y = Array4::<f32>::zeros((n, cout, hout, wout));

    for ni in 0..n {
        for oc in 0..cout {
            let bias = b.as_ref().map(|bb| bb[oc]).unwrap_or(0.0);

            for oh in 0..hout {
                let in_y0 = oh as isize * s - p;

                for ow in 0..wout {
                    let in_x0 = ow as isize * s - p;

                    let mut acc = bias;

                    for ic in 0..cin {
                        for r in 0..kh {
                            let iy = in_y0 + (r as isize) * d;
                            if iy < 0 || iy >= hin as isize {
                                continue;
                            }
                            let iyu = iy as usize;

                            for c in 0..kw {
                                let ix = in_x0 + (c as isize) * d;
                                if ix < 0 || ix >= win as isize {
                                    continue;
                                }
                                let ixu = ix as usize;

                                acc += x[(ni, ic, iyu, ixu)] * w[(oc, ic, r, c)];
                            }
                        }
                    }

                    y[(ni, oc, oh, ow)] = acc;
                }
            }
        }
    }

    Ok(y.into_dyn())
}
