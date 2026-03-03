use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::graph_form::nodes::unique_ids::Activation;

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 8;
const NR: usize = 8;

#[inline(always)]
unsafe fn micro_kernel_8x8_scalar(
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
    let mut acc = [[0.0f32; NR]; MR];

    unsafe {
        for p in 0..k {
            for (i, rows) in acc.iter_mut().enumerate().take(mr) {
                let a_val = *a.add(i * lda + p);
                for (j, val) in rows.iter_mut().enumerate().take(nr) {
                    *val += a_val * *b.add(p * ldb + j);
                }
            }
        }

        if accumulate {
            for (i, rows) in acc.iter().enumerate().take(mr) {
                for (j, value) in rows.iter().enumerate().take(nr) {
                    *c.add(i * ldc + j) += *value;
                }
            }
        } else {
            for (i, rows) in acc.iter().enumerate().take(mr) {
                for (j, value) in rows.iter().enumerate().take(nr) {
                    *c.add(i * ldc + j) = *value;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_8x8_avx2(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();
    let mut c2 = _mm256_setzero_ps();
    let mut c3 = _mm256_setzero_ps();
    let mut c4 = _mm256_setzero_ps();
    let mut c5 = _mm256_setzero_ps();
    let mut c6 = _mm256_setzero_ps();
    let mut c7 = _mm256_setzero_ps();

    unsafe {
        for p in 0..k {
            let b_row = _mm256_loadu_ps(b.add(p * ldb));

            let a0 = _mm256_broadcast_ss(&*a.add(p));
            let a1 = _mm256_broadcast_ss(&*a.add(lda + p));
            let a2 = _mm256_broadcast_ss(&*a.add(2 * lda + p));
            let a3 = _mm256_broadcast_ss(&*a.add(3 * lda + p));
            let a4 = _mm256_broadcast_ss(&*a.add(4 * lda + p));
            let a5 = _mm256_broadcast_ss(&*a.add(5 * lda + p));
            let a6 = _mm256_broadcast_ss(&*a.add(6 * lda + p));
            let a7 = _mm256_broadcast_ss(&*a.add(7 * lda + p));

            c0 = _mm256_fmadd_ps(a0, b_row, c0);
            c1 = _mm256_fmadd_ps(a1, b_row, c1);
            c2 = _mm256_fmadd_ps(a2, b_row, c2);
            c3 = _mm256_fmadd_ps(a3, b_row, c3);
            c4 = _mm256_fmadd_ps(a4, b_row, c4);
            c5 = _mm256_fmadd_ps(a5, b_row, c5);
            c6 = _mm256_fmadd_ps(a6, b_row, c6);
            c7 = _mm256_fmadd_ps(a7, b_row, c7);
        }

        if accumulate {
            let c0_old = _mm256_loadu_ps(c);
            let c1_old = _mm256_loadu_ps(c.add(ldc));
            let c2_old = _mm256_loadu_ps(c.add(2 * ldc));
            let c3_old = _mm256_loadu_ps(c.add(3 * ldc));
            let c4_old = _mm256_loadu_ps(c.add(4 * ldc));
            let c5_old = _mm256_loadu_ps(c.add(5 * ldc));
            let c6_old = _mm256_loadu_ps(c.add(6 * ldc));
            let c7_old = _mm256_loadu_ps(c.add(7 * ldc));

            _mm256_storeu_ps(c, _mm256_add_ps(c0_old, c0));
            _mm256_storeu_ps(c.add(ldc), _mm256_add_ps(c1_old, c1));
            _mm256_storeu_ps(c.add(2 * ldc), _mm256_add_ps(c2_old, c2));
            _mm256_storeu_ps(c.add(3 * ldc), _mm256_add_ps(c3_old, c3));
            _mm256_storeu_ps(c.add(4 * ldc), _mm256_add_ps(c4_old, c4));
            _mm256_storeu_ps(c.add(5 * ldc), _mm256_add_ps(c5_old, c5));
            _mm256_storeu_ps(c.add(6 * ldc), _mm256_add_ps(c6_old, c6));
            _mm256_storeu_ps(c.add(7 * ldc), _mm256_add_ps(c7_old, c7));
        } else {
            _mm256_storeu_ps(c, c0);
            _mm256_storeu_ps(c.add(ldc), c1);
            _mm256_storeu_ps(c.add(2 * ldc), c2);
            _mm256_storeu_ps(c.add(3 * ldc), c3);
            _mm256_storeu_ps(c.add(4 * ldc), c4);
            _mm256_storeu_ps(c.add(5 * ldc), c5);
            _mm256_storeu_ps(c.add(6 * ldc), c6);
            _mm256_storeu_ps(c.add(7 * ldc), c7);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_silu_and_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            let activated = silu_avx2(_mm256_add_ps(val, bias_v));
            _mm256_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_sigmoid_and_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            let activated = sigmoid_avx2(_mm256_add_ps(val, bias_v));
            _mm256_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn silu_avx2(x: __m256) -> __m256 {
    let left_margin = _mm256_set1_ps(-4.0);
    let right_margin = _mm256_set1_ps(4.0);
    let zeros = _mm256_setzero_ps();
    let quarter = _mm256_set1_ps(0.25);
    let one_over_eight = _mm256_set1_ps(0.125);
    let half = _mm256_set1_ps(0.5);

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // 0.25 * |x| * x * 0.125
    let part1 = _mm256_mul_ps(
        _mm256_mul_ps(quarter, _mm256_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm256_sub_ps(_mm256_add_ps(half, _mm256_mul_ps(quarter, x)), part1);

    let mut result = _mm256_mul_ps(x, part2);

    let mask_low = _mm256_cmp_ps(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm256_cmp_ps(x, right_margin, _CMP_GT_OQ);

    result = _mm256_blendv_ps(result, zeros, mask_low);

    result = _mm256_blendv_ps(result, x, mask_high);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);

    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            _mm256_storeu_ps(c.add(i), _mm256_add_ps(val, bias_v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn gemm_bias_blocked_avx2(
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
                            micro_kernel_8x8_avx2(
                                kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        } else {
                            micro_kernel_8x8_scalar(
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
                    apply_sigmoid_and_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_avx2(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_avx2(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_avx2(x: __m256) -> __m256 {
    let left_margin = _mm256_set1_ps(-4.0);
    let right_margin = _mm256_set1_ps(4.0);
    let zeros = _mm256_setzero_ps();
    let quarter = _mm256_set1_ps(0.25);
    let one_over_eight = _mm256_set1_ps(0.125);
    let half = _mm256_set1_ps(0.5);

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // 0.25 * |x| * x * 0.125
    let part1 = _mm256_mul_ps(
        _mm256_mul_ps(quarter, _mm256_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm256_sub_ps(_mm256_add_ps(half, _mm256_mul_ps(quarter, x)), part1);

    let mut result = _mm256_mul_ps(x, part2);

    let mask_low = _mm256_cmp_ps(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm256_cmp_ps(x, right_margin, _CMP_GT_OQ);

    result = _mm256_blendv_ps(result, zeros, mask_low);

    result = _mm256_blendv_ps(result, x, mask_high);

    //silu = x * sig <=> sig = silu / x
    result = _mm256_div_ps(result, x);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_silu_avx2(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(dst.add(i));
            let activated = silu_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_sigmoid_avx2(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(dst.add(i));
            let activated = sigmoid_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_silu_avx2_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(src.add(i));
            let activated = silu_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_sigmoid_avx2_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(src.add(i));
            let activated = sigmoid_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_add_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sub_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_sub_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mul_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn div_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_div_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) / *b.add(i);
        }
    }
}
