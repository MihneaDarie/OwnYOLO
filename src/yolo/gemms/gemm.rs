use crate::yolo::{gemms::mm512_gemm::gemm_bias_blocked_scalar, utils::silu};
use crate::yolo::gemms::mm256_gemm::gemm_bias_blocked_avx2;

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
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            unsafe {
                use crate::yolo::gemms::mm512_gemm::gemm_bias_blocked_avx512;

                gemm_bias_blocked_avx512(m, n, k, a, b, bias, c, use_silu);
            }
            return;
        }

        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                gemm_bias_blocked_avx2(m, n, k, a, b, bias, c, use_silu);
            }
            return;
        }
    }

    gemm_bias_blocked_scalar(m, n, k, a, b, bias, c, use_silu);
}
