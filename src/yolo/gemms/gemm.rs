#[cfg(target_arch = "x86_64")]
use crate::yolo::gemms::mm512_gemm::gemm_bias_blocked_avx512;
use crate::yolo::{
    context::appcontext::{Device, get_global_context},
    gemms::mm256_gemm::gemm_bias_blocked_avx2,
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
