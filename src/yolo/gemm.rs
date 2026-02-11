use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::yolo::utils::silu;

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 8;
const NR: usize = 8;

#[inline(always)]
unsafe fn micro_kernel_scalar(
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
            for i in 0..mr {
                let a_val = *a.add(i * lda + p);
                for j in 0..nr {
                    acc[i][j] += a_val * *b.add(p * ldb + j);
                }
            }
        }

        if accumulate {
            for i in 0..mr {
                for j in 0..nr {
                    *c.add(i * ldc + j) += acc[i][j];
                }
            }
        } else {
            for i in 0..mr {
                for j in 0..nr {
                    *c.add(i * ldc + j) = acc[i][j];
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

            let a0 = _mm256_broadcast_ss(&*a.add(0 * lda + p));
            let a1 = _mm256_broadcast_ss(&*a.add(1 * lda + p));
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
pub fn silu_avx2(x: __m256) -> __m256 {
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

pub fn sgemm_bias_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    use_silu: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    if m * n * k < 32 * 32 * 32 {
        for i in 0..m {
            let bias_val = bias.map(|bb| bb[i]).unwrap_or(0.0);
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = if use_silu {
                    silu(sum + bias_val)
                } else {
                    sum + bias_val
                };
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                gemm_bias_blocked_avx2(m, n, k, a, b, bias, c, use_silu);
            }
            return;
        }
    }

    gemm_bias_blocked_scalar(m, n, k, a, b, bias, c, use_silu);
}

fn gemm_bias_blocked_scalar(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    use_silu: bool,
) {
    let lda = k;
    let ldb = n;
    let ldc = n;
    let mt = (m + MC - 1) / MC;
    let nt = (n + NC - 1) / NC;

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

                        micro_kernel_scalar(
                            mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                        );
                    }
                }
            }
        }
    });

    match use_silu {
        true => match bias {
            Some(bb) => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    let bias_val = bb[i];
                    row.iter_mut().for_each(|v| *v = silu(*v + bias_val));
                });
            }
            None => {
                c.par_iter_mut().for_each(|v| *v = silu(*v));
            }
        },
        false => {
            if let Some(bb) = bias {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    let bias_val = bb[i];
                    row.iter_mut().for_each(|v| *v += bias_val);
                });
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn gemm_bias_blocked_avx2(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    use_silu: bool,
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

    let mt = (m + MC - 1) / MC;
    let nt = (n + NC - 1) / NC;

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
                            micro_kernel_scalar(
                                mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        }
                    }
                }
            }
        }
    });

    match use_silu {
        true => match bias {
            Some(bb) => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx2(row.as_mut_ptr(), n, bb[i]);
                });
            }
            None => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_silu_and_bias_avx2(row.as_mut_ptr(), n, 0.0);
                });
            }
        },
        false => {
            if let Some(bb) = bias {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx2(row.as_mut_ptr(), n, bb[i]);
                });
            }
        }
    }
}
