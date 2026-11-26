#include "bindings.h"
#include "helpers.cuh"
#include "proj.cuh"
#include "quat.cuh"
#include "quat_scale_to_covar_preci.cuh"
#include "transform.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Fused projection backward + sparse Adam (PennyGS, arXiv 2611.27525 Sec. 3.7)
 *
 * One thread per Gaussian of a single pinhole camera. A Gaussian the view does
 * not see (radius 0) reads its radius and returns: no gradient buffers and no
 * moments are touched. A visible Gaussian computes the gradients of its mean,
 * raw quaternion and log-scale exactly as `fully_fused_projection_bwd` does
 * (then v_log_scale = v_scale * scale) and immediately applies the selective
 * Adam step of `SelectiveAdam` (no bias correction) in registers. The first
 * moment is float32 or bfloat16 ("compact moments"); the second moment stays
 * float32: with beta2 = 0.999 its per-step change (1e-3 relative) is below half
 * a bfloat16 ulp (2^-9), so rounding it back would freeze it. The first moment
 * changes by (1 - beta1) = 10 % per step and stays within one ulp of float32.
 * The arithmetic is float32.
 ****************************************************************************/

template <typename M> inline __device__ float load_moment(const M *p) {
    return static_cast<float>(*p);
}
template <> inline __device__ float load_moment(const __nv_bfloat16 *p) {
    return __bfloat162float(*p);
}
template <typename M> inline __device__ void store_moment(M *p, const float v) {
    *p = static_cast<M>(v);
}
template <> inline __device__ void store_moment(__nv_bfloat16 *p, const float v) {
    *p = __float2bfloat16(v);
}

template <typename M, uint32_t D>
inline __device__ void adam_update(
    float *__restrict__ param,
    M *__restrict__ exp_avg,
    float *__restrict__ exp_avg_sq,
    const float (&grad)[D],
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < D; ++k) {
        const float m = b1 * load_moment(exp_avg + k) + (1.0f - b1) * grad[k];
        const float v = b2 * exp_avg_sq[k] + (1.0f - b2) * grad[k] * grad[k];
        param[k] += -lr * m / (sqrtf(v) + eps);
        store_moment(exp_avg + k, m);
        exp_avg_sq[k] = v;
    }
}

template <typename M>
__global__ void fused_projection_adam_kernel(
    const uint32_t N,
    float *__restrict__ means,      // [N, 3], updated in place
    float *__restrict__ quats,      // [N, 4] raw, updated in place
    float *__restrict__ log_scales, // [N, 3], updated in place
    const float *__restrict__ viewmat, // [4, 4]
    const float *__restrict__ K,       // [3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const int32_t *__restrict__ radii,  // [N]
    const float *__restrict__ conics,   // [N, 3]
    const float *__restrict__ v_means2d, // [N, 2]
    const float *__restrict__ v_depths,  // [N]
    const float *__restrict__ v_conics,  // [N, 3]
    M *__restrict__ means_m,
    float *__restrict__ means_v,
    M *__restrict__ quats_m,
    float *__restrict__ quats_v,
    M *__restrict__ scales_m,
    float *__restrict__ scales_v,
    const float lr_means,
    const float lr_quats,
    const float lr_scales,
    const float b1,
    const float b2,
    const float eps
) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0) {
        return;
    }
    conics += idx * 3;
    v_means2d += idx * 2;
    v_conics += idx * 3;

    // Same vector-Jacobian products as fully_fused_projection_bwd (pinhole).
    mat2<float> covar2d_inv = mat2<float>(conics[0], conics[1], conics[1], conics[2]);
    mat2<float> v_covar2d_inv = mat2<float>(
        v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]
    );
    mat2<float> v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    mat3<float> R = mat3<float>(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    vec3<float> t = vec3<float>(viewmat[3], viewmat[7], viewmat[11]);

    const vec3<float> mean = glm::make_vec3(means + idx * 3);
    const vec4<float> quat = glm::make_vec4(quats + idx * 4);
    const vec3<float> log_scale = glm::make_vec3(log_scales + idx * 3);
    const vec3<float> scale =
        vec3<float>(__expf(log_scale[0]), __expf(log_scale[1]), __expf(log_scale[2]));
    mat3<float> covar;
    quat_scale_to_covar_preci<float>(quat, scale, &covar, nullptr);
    vec3<float> mean_c;
    pos_world_to_cam(R, t, mean, mean_c);
    mat3<float> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    mat3<float> v_covar_c(0.f);
    vec3<float> v_mean_c(0.f);
    persp_proj_vjp<float>(
        mean_c, covar_c, K[0], K[4], K[2], K[5], image_width, image_height,
        v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c
    );
    v_mean_c.z += v_depths[idx];

    vec3<float> v_mean(0.f);
    mat3<float> v_covar(0.f);
    mat3<float> v_R(0.f);
    vec3<float> v_t(0.f);
    pos_world_to_cam_vjp(R, t, mean, v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    const mat3<float> rotmat = quat_to_rotmat<float>(quat);
    vec4<float> v_quat(0.f);
    vec3<float> v_scale(0.f);
    quat_scale_to_covar_vjp<float>(quat, scale, rotmat, v_covar, v_quat, v_scale);

    // Sparse Adam in registers.
    const float g_mean[3] = {v_mean[0], v_mean[1], v_mean[2]};
    const float g_quat[4] = {v_quat[0], v_quat[1], v_quat[2], v_quat[3]};
    const float g_scale[3] = {
        v_scale[0] * scale[0], v_scale[1] * scale[1], v_scale[2] * scale[2]
    };
    adam_update<M, 3>(
        means + idx * 3, means_m + idx * 3, means_v + idx * 3, g_mean, lr_means,
        b1, b2, eps
    );
    adam_update<M, 4>(
        quats + idx * 4, quats_m + idx * 4, quats_v + idx * 4, g_quat, lr_quats,
        b1, b2, eps
    );
    adam_update<M, 3>(
        log_scales + idx * 3, scales_m + idx * 3, scales_v + idx * 3, g_scale,
        lr_scales, b1, b2, eps
    );
}

void fused_projection_adam_tensor(
    torch::Tensor &means,      // [N, 3]
    torch::Tensor &quats,      // [N, 4]
    torch::Tensor &log_scales, // [N, 3]
    const torch::Tensor &viewmat, // [4, 4]
    const torch::Tensor &K,       // [3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const torch::Tensor &radii,     // [N]
    const torch::Tensor &conics,    // [N, 3]
    const torch::Tensor &v_means2d, // [N, 2]
    const torch::Tensor &v_depths,  // [N]
    const torch::Tensor &v_conics,  // [N, 3]
    torch::Tensor &means_m,
    torch::Tensor &means_v,
    torch::Tensor &quats_m,
    torch::Tensor &quats_v,
    torch::Tensor &scales_m,
    torch::Tensor &scales_v,
    const float lr_means,
    const float lr_quats,
    const float lr_scales,
    const float b1,
    const float b2,
    const float eps
) {
    GSPLAT_DEVICE_GUARD(means);
    for (const torch::Tensor &t :
         {means, quats, log_scales, viewmat, K, radii, conics, v_means2d,
          v_depths, v_conics, means_m, means_v, quats_m, quats_v, scales_m,
          scales_v}) {
        GSPLAT_CHECK_INPUT(t);
    }
    const uint32_t N = means.size(0);
    TORCH_CHECK(
        radii.numel() == N && conics.size(0) == N && v_means2d.size(0) == N,
        "fused_projection_adam: one camera, per-Gaussian [N, ...] inputs"
    );
    const auto moment_type = means_m.scalar_type();
    TORCH_CHECK(
        moment_type == torch::kFloat32 || moment_type == torch::kBFloat16,
        "first moments must be float32 or bfloat16"
    );
    TORCH_CHECK(
        means_v.scalar_type() == torch::kFloat32 &&
            quats_v.scalar_type() == torch::kFloat32 &&
            scales_v.scalar_type() == torch::kFloat32,
        "second moments must be float32: in bfloat16 the beta2 = 0.999 update "
        "is below half an ulp and the moment stops changing"
    );
    if (N == 0) {
        return;
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t blocks = (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;
#define __GS__FUSED_ADAM_(M, CPP)                                              \
    fused_projection_adam_kernel<M><<<blocks, GSPLAT_N_THREADS, 0, stream>>>(   \
        N, means.data_ptr<float>(), quats.data_ptr<float>(),                   \
        log_scales.data_ptr<float>(), viewmat.data_ptr<float>(),               \
        K.data_ptr<float>(), image_width, image_height,                        \
        radii.data_ptr<int32_t>(), conics.data_ptr<float>(),                   \
        v_means2d.data_ptr<float>(), v_depths.data_ptr<float>(),               \
        v_conics.data_ptr<float>(),                                            \
        reinterpret_cast<M *>(means_m.data_ptr<CPP>()),                        \
        means_v.data_ptr<float>(),                                             \
        reinterpret_cast<M *>(quats_m.data_ptr<CPP>()),                        \
        quats_v.data_ptr<float>(),                                             \
        reinterpret_cast<M *>(scales_m.data_ptr<CPP>()),                       \
        scales_v.data_ptr<float>(), lr_means, lr_quats, lr_scales, b1, b2,     \
        eps                                                                    \
    )
    if (moment_type == torch::kBFloat16) {
        __GS__FUSED_ADAM_(__nv_bfloat16, at::BFloat16);
    } else {
        __GS__FUSED_ADAM_(float, float);
    }
#undef __GS__FUSED_ADAM_
}

} // namespace gsplat
