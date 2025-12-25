use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const MC: usize = 128;
const KC: usize = 256;
const NC: usize = 512;

const MR: usize = 8;
const NR: usize = 8;

pub fn sgemm_parallel(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    if beta == 0.0 {
        c.par_iter_mut().for_each(|x| *x = 0.0);
    } else if beta != 1.0 {
        c.par_iter_mut().for_each(|x| *x *= beta);
    }

    let row_tiles: Vec<usize> = (0..m).step_by(MC).collect();

    row_tiles.into_par_iter().for_each(|i0| {
        let i_end = (i0 + MC).min(m);
        let tile_m = i_end - i0;

        for j0 in (0..n).step_by(NC) {
            let j_end = (j0 + NC).min(n);
            let tile_n = j_end - j0;

            for p0 in (0..k).step_by(KC) {
                let p_end = (p0 + KC).min(k);
                let tile_k = p_end - p0;

                for i in (0..tile_m).step_by(MR) {
                    let mr = (tile_m - i).min(MR);

                    for j in (0..tile_n).step_by(NR) {
                        let nr = (tile_n - j).min(NR);

                        let a_ptr = &a[(i0 + i) * k + p0..];
                        let b_ptr = &b[p0 * n + (j0 + j)..];

                        let c_offset = (i0 + i) * n + (j0 + j);

                        unsafe {
                            let c_ptr = c.as_ptr().add(c_offset) as *mut f32;
                            micro_kernel(mr, nr, tile_k, alpha, a_ptr, k, b_ptr, n, c_ptr, n);
                        }
                    }
                }
            }
        }
    });
}

#[inline(always)]
unsafe fn micro_kernel(
    mr: usize,
    nr: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if mr == MR
            && nr == NR
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            micro_kernel_8x8_avx2(k, alpha, a, lda, b, ldb, c, ldc);
            return;
        }

        micro_kernel_scalar(mr, nr, k, alpha, a, lda, b, ldb, c, ldc);
    }
}

#[inline(always)]
unsafe fn micro_kernel_scalar(
    mr: usize,
    nr: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];

    unsafe {
        for p in 0..k {
            for i in 0..mr {
                let a_val = *a.get_unchecked(i * lda + p);
                for j in 0..nr {
                    let b_val = *b.get_unchecked(p * ldb + j);
                    acc[i][j] += a_val * b_val;
                }
            }
        }

        for i in 0..mr {
            for j in 0..nr {
                let c_ptr = c.add(i * ldc + j);
                *c_ptr += alpha * acc[i][j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_8x8_avx2(
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
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

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    unsafe {
        for p in 0..k {
            let b_row = _mm256_loadu_ps(b_ptr.add(p * ldb));

            let a0 = _mm256_set1_ps(*a_ptr.add(0 * lda + p));
            let a1 = _mm256_set1_ps(*a_ptr.add(1 * lda + p));
            let a2 = _mm256_set1_ps(*a_ptr.add(2 * lda + p));
            let a3 = _mm256_set1_ps(*a_ptr.add(3 * lda + p));
            let a4 = _mm256_set1_ps(*a_ptr.add(4 * lda + p));
            let a5 = _mm256_set1_ps(*a_ptr.add(5 * lda + p));
            let a6 = _mm256_set1_ps(*a_ptr.add(6 * lda + p));
            let a7 = _mm256_set1_ps(*a_ptr.add(7 * lda + p));

            c0 = _mm256_fmadd_ps(a0, b_row, c0);
            c1 = _mm256_fmadd_ps(a1, b_row, c1);
            c2 = _mm256_fmadd_ps(a2, b_row, c2);
            c3 = _mm256_fmadd_ps(a3, b_row, c3);
            c4 = _mm256_fmadd_ps(a4, b_row, c4);
            c5 = _mm256_fmadd_ps(a5, b_row, c5);
            c6 = _mm256_fmadd_ps(a6, b_row, c6);
            c7 = _mm256_fmadd_ps(a7, b_row, c7);
        }

        let alpha_vec = _mm256_set1_ps(alpha);

        let c0_old = _mm256_loadu_ps(c.add(0 * ldc));
        let c1_old = _mm256_loadu_ps(c.add(1 * ldc));
        let c2_old = _mm256_loadu_ps(c.add(2 * ldc));
        let c3_old = _mm256_loadu_ps(c.add(3 * ldc));
        let c4_old = _mm256_loadu_ps(c.add(4 * ldc));
        let c5_old = _mm256_loadu_ps(c.add(5 * ldc));
        let c6_old = _mm256_loadu_ps(c.add(6 * ldc));
        let c7_old = _mm256_loadu_ps(c.add(7 * ldc));

        _mm256_storeu_ps(c.add(0 * ldc), _mm256_fmadd_ps(alpha_vec, c0, c0_old));
        _mm256_storeu_ps(c.add(1 * ldc), _mm256_fmadd_ps(alpha_vec, c1, c1_old));
        _mm256_storeu_ps(c.add(2 * ldc), _mm256_fmadd_ps(alpha_vec, c2, c2_old));
        _mm256_storeu_ps(c.add(3 * ldc), _mm256_fmadd_ps(alpha_vec, c3, c3_old));
        _mm256_storeu_ps(c.add(4 * ldc), _mm256_fmadd_ps(alpha_vec, c4, c4_old));
        _mm256_storeu_ps(c.add(5 * ldc), _mm256_fmadd_ps(alpha_vec, c5, c5_old));
        _mm256_storeu_ps(c.add(6 * ldc), _mm256_fmadd_ps(alpha_vec, c6, c6_old));
        _mm256_storeu_ps(c.add(7 * ldc), _mm256_fmadd_ps(alpha_vec, c7, c7_old));
    }
}
