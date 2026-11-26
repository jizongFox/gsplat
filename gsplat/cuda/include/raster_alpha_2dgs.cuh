#ifndef GSPLAT_RASTER_ALPHA_2DGS_CUH
#define GSPLAT_RASTER_ALPHA_2DGS_CUH

#include "2dgs.cuh"
#include "raster_alpha.cuh"
#include "types.cuh"

namespace gsplat {

// Per-pixel alpha of the 2DGS pixel rasterizer, shared by the forward kernels in
// rasterize_to_pixels_2dgs_fwd.cu (default and subtile), with the thresholds of
// raster_alpha.cuh. The backward in rasterize_to_pixels_2dgs_bwd.cu repeats it.
//
// The weight is the smaller of the ray-splat intersection |s|^2 in the splat's
// (u, v) frame (u_M, v_M, w_M: rows of ray_transforms) and the screen-space
// low-pass FILTER_INV_SQUARE |p - mean2d|^2; alpha is clamped to MAX_ALPHA.
// Returns false where the surfel is skipped: a ray parallel to the splat
// (ray_cross.z == 0), negative sigma, or alpha below ALPHA_THRESHOLD.
template <typename S>
inline __device__ bool surfel_alpha(
    const vec3<S> xy_opac,
    const vec3<S> u_M,
    const vec3<S> v_M,
    const vec3<S> w_M,
    const S px,
    const S py,
    S &alpha
) {
    // homogeneous planes of the pixel's ray in the splat's (u, v) frame; their
    // cross product is the intersection
    const vec3<S> h_u = px * w_M - u_M;
    const vec3<S> h_v = py * w_M - v_M;
    const vec3<S> ray_cross = glm::cross(h_u, h_v);
    if (ray_cross.z == 0.0) {
        return false;
    }
    const vec2<S> s =
        vec2<S>(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);
    const S gauss_weight_3d = s.x * s.x + s.y * s.y;
    const vec2<S> d = {xy_opac.x - px, xy_opac.y - py};
    const S gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
    const S gauss_weight = min(gauss_weight_3d, gauss_weight_2d);
    const S sigma = 0.5f * gauss_weight;
    alpha = min(MAX_ALPHA, xy_opac.z * __expf(-sigma));
    return !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
}

// Transmittance after blending a surfel of this alpha.
template <typename S> inline __device__ S surfel_next_T(const S T, const S alpha) {
    return __fmul_rn(T, __fadd_rn(1.0f, -alpha));
}

// One blending step for an accepted surfel: colours, normals, distortion and
// median depth. Every rounding is explicit: the depth (the last colour channel)
// times the blending weight appears in both the colour and the distortion
// sums, and leaving it to the compiler lets it fuse one and reuse the other
// differently per kernel. The kernels sharing this step therefore agree bit for
// bit.
template <uint32_t COLOR_DIM, typename S>
inline __device__ void surfel_blend(
    const S alpha,
    const S T,
    const S *__restrict__ c,
    const S *__restrict__ n,
    S (&pix_out)[COLOR_DIM],
    S (&normal_out)[3],
    const bool with_distort,
    S &distort,
    S &accum_vis_depth,
    S &median_depth,
    uint32_t &median_idx,
    const uint32_t isect_idx
) {
    const S vis = __fmul_rn(T, alpha);
#pragma unroll
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        pix_out[k] = __fmaf_rn(vis, c[k], pix_out[k]);
    }
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        normal_out[k] = __fmaf_rn(vis, n[k], normal_out[k]);
    }
    const S depth = c[COLOR_DIM - 1];
    if (with_distort) {
        // nerfacc: loss_bi_0 = weights * t_mids * exclusive_sum(weights),
        // loss_bi_1 = weights * exclusive_sum(weights * t_mids)
        const S vis_depth = __fmul_rn(vis, depth);
        const S distort_bi_1 = __fmul_rn(accum_vis_depth, vis);
        const S diff = __fmaf_rn(vis_depth, __fadd_rn(1.0f, -T), -distort_bi_1);
        distort = __fmaf_rn(diff, 2.0f, distort);
        accum_vis_depth = __fadd_rn(accum_vis_depth, vis_depth);
    }
    // median depth: the surfel that brings the accumulated opacity over 0.5
    if (T > 0.5f) {
        median_depth = depth;
        median_idx = isect_idx;
    }
}

// Outputs of a pixel no surfel reaches (a masked-out tile): the background
// colour (0 without backgrounds), alpha 0, normal 0, distortion 0, median depth
// 0, last_ids 0 and median_ids 0, the values an unmasked empty pixel gets.
template <uint32_t COLOR_DIM, typename S>
inline __device__ void write_empty_surfel_pixel(
    const int32_t pix_id,
    const S *__restrict__ backgrounds,
    S *__restrict__ render_colors,
    S *__restrict__ render_alphas,
    S *__restrict__ render_normals,
    S *__restrict__ render_distort,
    S *__restrict__ render_median,
    int32_t *__restrict__ last_ids,
    int32_t *__restrict__ median_ids
) {
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        render_colors[pix_id * COLOR_DIM + k] =
            backgrounds == nullptr ? 0.0f : backgrounds[k];
    }
    render_alphas[pix_id] = 0.0f;
    for (uint32_t k = 0; k < 3; ++k) {
        render_normals[pix_id * 3 + k] = 0.0f;
    }
    if (render_distort != nullptr) {
        render_distort[pix_id] = 0.0f;
    }
    render_median[pix_id] = 0.0f;
    last_ids[pix_id] = 0;
    median_ids[pix_id] = 0;
}

} // namespace gsplat

#endif // GSPLAT_RASTER_ALPHA_2DGS_CUH
