use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::yolo::utils::silu_f32;
#[cfg(target_arch = "x86_64")]
use crate::{graph_form::nodes::unique_ids::Activation, yolo::utils::aprox_sigmoid_f32};

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 16;
const NR: usize = 16;

#[inline(always)]
unsafe fn micro_kernel_16x16_scalar(
    mr: usize,
    nr: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    let mut acc = [0.0f32; 256];

    unsafe {
        for p in 0..k {
            for i in 0..mr {
                let val = *a.add(i * lda + p);
                for j in 0..nr {
                    acc[i * 16 + j] += val * *b.add(p * ldb + j);
                }
            }
        }
        for i in 0..mr {
            for j in 0..nr {
                if accumulate {
                    *c.add(i * ldc + j) += acc[i * 16 + j];
                } else {
                    *c.add(i * ldc + j) = acc[i * 16 + j]
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn micro_kernel_16x16_avx_512(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();
    let mut c6 = _mm512_setzero_ps();
    let mut c7 = _mm512_setzero_ps();
    let mut c8 = _mm512_setzero_ps();
    let mut c9 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c12 = _mm512_setzero_ps();
    let mut c13 = _mm512_setzero_ps();
    let mut c14 = _mm512_setzero_ps();
    let mut c15 = _mm512_setzero_ps();

    unsafe {
        for p in 0..k {
            let b = _mm512_loadu_ps(b.add(p * ldb));

            let a0 = _mm512_set1_ps(*a.add(p));
            let a1 = _mm512_set1_ps(*a.add(lda + p));
            let a2 = _mm512_set1_ps(*a.add(2 * lda + p));
            let a3 = _mm512_set1_ps(*a.add(3 * lda + p));
            let a4 = _mm512_set1_ps(*a.add(4 * lda + p));
            let a5 = _mm512_set1_ps(*a.add(5 * lda + p));
            let a6 = _mm512_set1_ps(*a.add(6 * lda + p));
            let a7 = _mm512_set1_ps(*a.add(7 * lda + p));
            let a8 = _mm512_set1_ps(*a.add(8 * lda + p));
            let a9 = _mm512_set1_ps(*a.add(9 * lda + p));
            let a10 = _mm512_set1_ps(*a.add(10 * lda + p));
            let a11 = _mm512_set1_ps(*a.add(11 * lda + p));
            let a12 = _mm512_set1_ps(*a.add(12 * lda + p));
            let a13 = _mm512_set1_ps(*a.add(13 * lda + p));
            let a14 = _mm512_set1_ps(*a.add(14 * lda + p));
            let a15 = _mm512_set1_ps(*a.add(15 * lda + p));

            c0 = _mm512_fmadd_ps(a0, b, c0);
            c1 = _mm512_fmadd_ps(a1, b, c1);
            c2 = _mm512_fmadd_ps(a2, b, c2);
            c3 = _mm512_fmadd_ps(a3, b, c3);
            c4 = _mm512_fmadd_ps(a4, b, c4);
            c5 = _mm512_fmadd_ps(a5, b, c5);
            c6 = _mm512_fmadd_ps(a6, b, c6);
            c7 = _mm512_fmadd_ps(a7, b, c7);
            c8 = _mm512_fmadd_ps(a8, b, c8);
            c9 = _mm512_fmadd_ps(a9, b, c9);
            c10 = _mm512_fmadd_ps(a10, b, c10);
            c11 = _mm512_fmadd_ps(a11, b, c11);
            c12 = _mm512_fmadd_ps(a12, b, c12);
            c13 = _mm512_fmadd_ps(a13, b, c13);
            c14 = _mm512_fmadd_ps(a14, b, c14);
            c15 = _mm512_fmadd_ps(a15, b, c15);
        }

        if accumulate {
            let c0_old = _mm512_loadu_ps(c);
            let c1_old = _mm512_loadu_ps(c.add(ldc));
            let c2_old = _mm512_loadu_ps(c.add(2 * ldc));
            let c3_old = _mm512_loadu_ps(c.add(3 * ldc));
            let c4_old = _mm512_loadu_ps(c.add(4 * ldc));
            let c5_old = _mm512_loadu_ps(c.add(5 * ldc));
            let c6_old = _mm512_loadu_ps(c.add(6 * ldc));
            let c7_old = _mm512_loadu_ps(c.add(7 * ldc));
            let c8_old = _mm512_loadu_ps(c.add(8 * ldc));
            let c9_old = _mm512_loadu_ps(c.add(9 * ldc));
            let c10_old = _mm512_loadu_ps(c.add(10 * ldc));
            let c11_old = _mm512_loadu_ps(c.add(11 * ldc));
            let c12_old = _mm512_loadu_ps(c.add(12 * ldc));
            let c13_old = _mm512_loadu_ps(c.add(13 * ldc));
            let c14_old = _mm512_loadu_ps(c.add(14 * ldc));
            let c15_old = _mm512_loadu_ps(c.add(15 * ldc));

            _mm512_storeu_ps(c, _mm512_add_ps(c0_old, c0));
            _mm512_storeu_ps(c.add(ldc), _mm512_add_ps(c1_old, c1));
            _mm512_storeu_ps(c.add(2 * ldc), _mm512_add_ps(c2_old, c2));
            _mm512_storeu_ps(c.add(3 * ldc), _mm512_add_ps(c3_old, c3));
            _mm512_storeu_ps(c.add(4 * ldc), _mm512_add_ps(c4_old, c4));
            _mm512_storeu_ps(c.add(5 * ldc), _mm512_add_ps(c5_old, c5));
            _mm512_storeu_ps(c.add(6 * ldc), _mm512_add_ps(c6_old, c6));
            _mm512_storeu_ps(c.add(7 * ldc), _mm512_add_ps(c7_old, c7));
            _mm512_storeu_ps(c.add(8 * ldc), _mm512_add_ps(c8_old, c8));
            _mm512_storeu_ps(c.add(9 * ldc), _mm512_add_ps(c9_old, c9));
            _mm512_storeu_ps(c.add(10 * ldc), _mm512_add_ps(c10_old, c10));
            _mm512_storeu_ps(c.add(11 * ldc), _mm512_add_ps(c11_old, c11));
            _mm512_storeu_ps(c.add(12 * ldc), _mm512_add_ps(c12_old, c12));
            _mm512_storeu_ps(c.add(13 * ldc), _mm512_add_ps(c13_old, c13));
            _mm512_storeu_ps(c.add(14 * ldc), _mm512_add_ps(c14_old, c14));
            _mm512_storeu_ps(c.add(15 * ldc), _mm512_add_ps(c15_old, c15));
        } else {
            _mm512_storeu_ps(c, c0);
            _mm512_storeu_ps(c.add(ldc), c1);
            _mm512_storeu_ps(c.add(2 * ldc), c2);
            _mm512_storeu_ps(c.add(3 * ldc), c3);
            _mm512_storeu_ps(c.add(4 * ldc), c4);
            _mm512_storeu_ps(c.add(5 * ldc), c5);
            _mm512_storeu_ps(c.add(6 * ldc), c6);
            _mm512_storeu_ps(c.add(7 * ldc), c7);
            _mm512_storeu_ps(c.add(8 * ldc), c8);
            _mm512_storeu_ps(c.add(9 * ldc), c9);
            _mm512_storeu_ps(c.add(10 * ldc), c10);
            _mm512_storeu_ps(c.add(11 * ldc), c11);
            _mm512_storeu_ps(c.add(12 * ldc), c12);
            _mm512_storeu_ps(c.add(13 * ldc), c13);
            _mm512_storeu_ps(c.add(14 * ldc), c14);
            _mm512_storeu_ps(c.add(15 * ldc), c15);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_silu_and_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            let activated = silu_avx512(_mm512_add_ps(val, bias_v));
            _mm512_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_sigmoid_and_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            let activated = sigmoid_avx512(_mm512_add_ps(val, bias_v));
            _mm512_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            _mm512_storeu_ps(c.add(i), _mm512_add_ps(val, bias_v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn silu_avx512(x: __m512) -> __m512 {
    let left_margin = _mm512_set1_ps(-4.0);
    let right_margin = _mm512_set1_ps(4.0);
    let zeros = _mm512_setzero_ps();
    let quarter = _mm512_set1_ps(0.25);
    let one_over_eight = _mm512_set1_ps(0.125);
    let half = _mm512_set1_ps(0.5);

    let abs_x = unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x) };

    // 0.25 * |x| * x * 0.125
    let part1 = _mm512_mul_ps(
        _mm512_mul_ps(quarter, _mm512_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm512_sub_ps(_mm512_add_ps(half, _mm512_mul_ps(quarter, x)), part1);

    let mut result = _mm512_mul_ps(x, part2);

    let mask_low = _mm512_cmp_ps_mask(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm512_cmp_ps_mask(x, right_margin, _CMP_GT_OQ);

    result = _mm512_mask_mov_ps(result, mask_low, zeros);
    result = _mm512_mask_mov_ps(result, mask_high, x);

    result
}

pub fn gemm_bias_blocked_scalar(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    activation: Activation,
) {
    let lda = k;
    let ldb = n;
    let ldc = n;
    let mt = m.div_ceil(MC);
    let nt = n.div_ceil(NC);

    let a_base = a.as_ptr() as usize;
    let b_base = b.as_ptr() as usize;
    let c_base = c.as_mut_ptr() as usize;

    (0..mt * nt).into_par_iter().for_each(|t| {
        let a_ptr_base = a_base as *const f32;
        let b_ptr_base = b_base as *const f32;
        let c_ptr_base = c_base as *mut f32;

        let i0 = (t / nt) * MC;
        let j0 = (t % nt) * NC;

        let mc = (m - i0).min(MC);
        let nc = (n - j0).min(NC);

        for p0 in (0..k).step_by(KC) {
            let kc = (k - p0).min(KC);
            let accumulate = p0 != 0;

            for i in (0..mc).step_by(MR) {
                let mr = (mc - i).min(MR);

                for j in (0..nc).step_by(NR) {
                    let nr = (nc - j).min(NR);

                    unsafe {
                        let a_ptr = a_ptr_base.add((i0 + i) * lda + p0);
                        let b_ptr = b_ptr_base.add(p0 * ldb + (j0 + j));
                        let c_ptr = c_ptr_base.add((i0 + i) * ldc + (j0 + j));

                        micro_kernel_16x16_scalar(
                            mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                        );
                    }
                }
            }
        }
    });
    match bias {
        Some(bias) => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn gemm_bias_blocked_avx512(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    activation: Activation,
) {
    if let Some(bb) = bias {
        debug_assert_eq!(bb.len(), m);
    }

    let lda = k;
    let ldb = n;
    let ldc = n;

    let a_base = a.as_ptr() as usize;
    let b_base = b.as_ptr() as usize;
    let c_base = c.as_mut_ptr() as usize;

    let mt = m.div_ceil(MC);
    let nt = n.div_ceil(NC);

    (0..mt * nt).into_par_iter().for_each(|t| {
        let a_ptr_base = a_base as *const f32;
        let b_ptr_base = b_base as *const f32;
        let c_ptr_base = c_base as *mut f32;

        let i0 = (t / nt) * MC;
        let j0 = (t % nt) * NC;

        let mc = (m - i0).min(MC);
        let nc = (n - j0).min(NC);

        for p0 in (0..k).step_by(KC) {
            let kc = (k - p0).min(KC);
            let accumulate = p0 != 0;

            for i in (0..mc).step_by(MR) {
                let mr = (mc - i).min(MR);

                for j in (0..nc).step_by(NR) {
                    let nr = (nc - j).min(NR);

                    unsafe {
                        let a_ptr = a_ptr_base.add((i0 + i) * lda + p0);
                        let b_ptr = b_ptr_base.add(p0 * ldb + (j0 + j));
                        let c_ptr = c_ptr_base.add((i0 + i) * ldc + (j0 + j));

                        if mr == MR && nr == NR {
                            micro_kernel_16x16_avx_512(
                                kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        } else {
                            micro_kernel_16x16_scalar(
                                mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        }
                    }
                }
            }
        }
    });

    match bias {
        Some(bias) => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sigmoid_avx512(x: __m512) -> __m512 {
    let left_margin = _mm512_set1_ps(-4.0);
    let right_margin = _mm512_set1_ps(4.0);
    let zeros = _mm512_setzero_ps();
    let quarter = _mm512_set1_ps(0.25);
    let one_over_eight = _mm512_set1_ps(0.125);
    let half = _mm512_set1_ps(0.5);

    let abs_x = unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x) };

    // 0.25 * |x| * x * 0.125
    let part1 = _mm512_mul_ps(
        _mm512_mul_ps(quarter, _mm512_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm512_sub_ps(_mm512_add_ps(half, _mm512_mul_ps(quarter, x)), part1);

    let mut result = _mm512_mul_ps(x, part2);

    let mask_low = _mm512_cmp_ps_mask(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm512_cmp_ps_mask(x, right_margin, _CMP_GT_OQ);

    result = _mm512_mask_mov_ps(result, mask_low, zeros);
    result = _mm512_mask_mov_ps(result, mask_high, x);

    result = _mm512_div_ps(result, x);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_silu_avx512(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(dst.add(i));
            let activated = silu_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_sigmoid_avx512(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(dst.add(i));
            let activated = sigmoid_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_silu_avx512_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let val = _mm512_loadu_ps(src.add(i));
            let activated = silu_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
        for i in chunks..n {
            let x = *src.add(i);
            *dst.add(i) = silu_f32(x);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_sigmoid_avx512_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(src.add(i));
            let activated = sigmoid_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn add_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_add_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sub_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_sub_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) - *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn mul_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_mul_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) * *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn div_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_div_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) / *b.add(i);
        }
    }
}
