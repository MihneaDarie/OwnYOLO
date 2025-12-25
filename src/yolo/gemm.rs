use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 8;
const NR: usize = 8;

#[inline(always)]
unsafe fn micro_kernel_accum(
    mr: usize,
    nr: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if mr == MR
            && nr == NR
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                micro_kernel_8x8_avx2_accum(k, alpha, a, lda, b, ldb, c, ldc);
            }
            return;
        }
    }
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
    }

    unsafe {
        for i in 0..mr {
            for j in 0..nr {
                *c.add(i * ldc + j) += alpha * acc[i][j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_8x8_avx2_accum(
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
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

        let alpha_v = _mm256_set1_ps(alpha);

        let c0_old = _mm256_loadu_ps(c);
        let c1_old = _mm256_loadu_ps(c.add(ldc));
        let c2_old = _mm256_loadu_ps(c.add(2 * ldc));
        let c3_old = _mm256_loadu_ps(c.add(3 * ldc));
        let c4_old = _mm256_loadu_ps(c.add(4 * ldc));
        let c5_old = _mm256_loadu_ps(c.add(5 * ldc));
        let c6_old = _mm256_loadu_ps(c.add(6 * ldc));
        let c7_old = _mm256_loadu_ps(c.add(7 * ldc));

        _mm256_storeu_ps(c, _mm256_fmadd_ps(alpha_v, c0, c0_old));
        _mm256_storeu_ps(c.add(ldc), _mm256_fmadd_ps(alpha_v, c1, c1_old));
        _mm256_storeu_ps(c.add(2 * ldc), _mm256_fmadd_ps(alpha_v, c2, c2_old));
        _mm256_storeu_ps(c.add(3 * ldc), _mm256_fmadd_ps(alpha_v, c3, c3_old));
        _mm256_storeu_ps(c.add(4 * ldc), _mm256_fmadd_ps(alpha_v, c4, c4_old));
        _mm256_storeu_ps(c.add(5 * ldc), _mm256_fmadd_ps(alpha_v, c5, c5_old));
        _mm256_storeu_ps(c.add(6 * ldc), _mm256_fmadd_ps(alpha_v, c6, c6_old));
        _mm256_storeu_ps(c.add(7 * ldc), _mm256_fmadd_ps(alpha_v, c7, c7_old));
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
            let bias_val = bias.map(|b| b[i]).unwrap_or(0.0);
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                match use_silu {
                    true => {
                        c[i * n + j] = silu(sum + bias_val);
                    }
                    false => {
                        c[i * n + j] = sum + bias_val;
                    }
                }
            }
        }
        return;
    }

    gemm_bias_blocked(m, n, k, a, b, bias, c, use_silu);
}

fn gemm_bias_blocked(
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

    c.iter_mut().for_each(|x| *x = 0.0);

    let m_tiles: Vec<usize> = (0..m).step_by(MC).collect();

    m_tiles.into_par_iter().for_each(|i0| {
        let mc = (m - i0).min(MC);

        for j0 in (0..n).step_by(NC) {
            let nc = (n - j0).min(NC);

            for p0 in (0..k).step_by(KC) {
                let kc = (k - p0).min(KC);

                for i in (0..mc).step_by(MR) {
                    let mr = (mc - i).min(MR);

                    for j in (0..nc).step_by(NR) {
                        let nr = (nc - j).min(NR);

                        let a_ptr = &a[(i0 + i) * lda + p0..];
                        let b_ptr = &b[p0 * ldb + (j0 + j)..];

                        unsafe {
                            let c_ptr = c.as_ptr().add((i0 + i) * ldc + (j0 + j)) as *mut f32;
                            micro_kernel_accum(
                                mr,
                                nr,
                                kc,
                                1.0,
                                a_ptr.as_ptr(),
                                lda,
                                b_ptr.as_ptr(),
                                ldb,
                                c_ptr,
                                ldc,
                            );
                        }
                    }
                }
            }
        }
    });

    match use_silu {
        true => match bias {
            Some(b) => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    let bias_val = b[i];
                    row.iter_mut().for_each(|v| *v = silu(*v + bias_val));
                });
            }
            None => {
                c.par_iter_mut().for_each(|v| *v = silu(*v));
            }
        },
        false => {
            if let Some(b) = bias {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    let bias_val = b[i];
                    row.iter_mut().for_each(|v| *v += bias_val);
                });
            }
        }
    }
}
