use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator, IntoParallelRefMutIterator,
    ParallelIterator as _,
};

#[cfg(target_arch = "x86_64")]
use crate::yolo::gemms::mm512_gemm::gemm_bias_blocked_avx512;
use crate::yolo::{
    context::appcontext::{Device, get_global_context},
    gemms::{
        mm256_gemm::{
            add_avx2, apply_sigmoid_avx2, apply_silu_avx2, div_avx2, gemm_bias_blocked_avx2,
            mul_avx2, sub_avx2,
        },
        mm512_gemm::{
            add_avx512, apply_sigmoid_avx512, apply_silu_avx512, div_avx512, mul_avx512, sub_avx512,
        },
    },
    utils::aprox_sigmoid_f32,
};
use crate::yolo::{gemms::mm512_gemm::gemm_bias_blocked_scalar, utils::silu_f32};

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
                    silu_f32(sum + bias_val)
                } else {
                    sum + bias_val
                };
            }
        }
        return;
    }

    let context = get_global_context();

    if context.get_device() == Device::Cpu {
        match context.get_gemm_type() {
            crate::yolo::context::appcontext::GemmType::Avx2 => {
                unsafe { gemm_bias_blocked_avx2(m, n, k, a, b, bias, c, use_silu) };
            }
            crate::yolo::context::appcontext::GemmType::Avx512 => {
                unsafe { gemm_bias_blocked_avx512(m, n, k, a, b, bias, c, use_silu) };
            }
            _ => {
                gemm_bias_blocked_scalar(m, n, k, a, b, bias, c, use_silu);
            }
        }
    } else {
    }
}

pub fn apply_silu(dst: &mut [f32], src: &[f32], n: usize) {
    let context = get_global_context();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = dst.as_ptr();

    match context.get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { apply_silu_avx2(dst_ptr, src_ptr, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => unsafe {
            apply_silu_avx512(dst_ptr, src_ptr, n);
        },
        _ => {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = aprox_sigmoid_f32(*s));
        }
    }
}

pub fn apply_sigmoid(dst: &mut [f32], src: &[f32], n: usize) {
    let context = get_global_context();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    match context.get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { apply_sigmoid_avx2(dst_ptr, src_ptr, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => unsafe {
            apply_sigmoid_avx512(dst_ptr, src_ptr, n);
        },
        _ => {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = aprox_sigmoid_f32(*s));
        }
    }
}

pub fn add_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32], n: usize) {
    let dst_ptr_mut = dst.as_mut_ptr();
    let sa_ptr = a.as_ptr();
    let sb_ptr = b.as_ptr();

    match get_global_context().get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { add_avx2(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => {
            unsafe { add_avx512(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        _ => {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a, b))| *d = *a + *b);
        }
    }
}

pub fn sub_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32], n: usize) {
    let dst_ptr_mut = dst.as_mut_ptr();
    let sa_ptr = a.as_ptr();
    let sb_ptr = b.as_ptr();

    match get_global_context().get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { sub_avx2(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => {
            unsafe { sub_avx512(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        _ => {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a, b))| *d = *a - *b);
        }
    }
}

pub fn mul_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32], n: usize) {
    let dst_ptr_mut = dst.as_mut_ptr();
    let sa_ptr = a.as_ptr();
    let sb_ptr = b.as_ptr();

    match get_global_context().get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { mul_avx2(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => {
            unsafe { mul_avx512(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        _ => {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a, b))| *d = *a * *b);
        }
    }
}

pub fn div_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32], n: usize) {
    let dst_ptr_mut = dst.as_mut_ptr();
    let sa_ptr = a.as_ptr();
    let sb_ptr = b.as_ptr();

    match get_global_context().get_gemm_type() {
        crate::yolo::context::appcontext::GemmType::Avx2 => {
            unsafe { div_avx2(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        crate::yolo::context::appcontext::GemmType::Avx512 => {
            unsafe { div_avx512(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        _ => {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a, b))| *d = *a / *b);
        }
    }
}
