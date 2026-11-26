#ifndef GSPLAT_RASTERIZE_TO_PIXELS_2DGS_FWD_SUBTILE_CUH
#define GSPLAT_RASTERIZE_TO_PIXELS_2DGS_FWD_SUBTILE_CUH

// The 2DGS subtile forward kernel and its launch. Private to
// rasterize_to_pixels_2dgs_fwd.cu, which includes it (same translation unit).

#include "2dgs.cuh"
#include "bindings.h"
#include "raster_alpha_2dgs.cuh"
#include "subtile_layout.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * 2DGS forward with subtile masks (the 2DGS counterpart of
 * rasterize_to_pixels_fwd_subtile.cuh)
 *
 * When a batch of surfels is staged, each surfel also records which 8x4
 * subtiles of the tile it may reach; each warp compacts the entries reaching
 * its subtile into a hit list, in depth order, and runs the default kernel's
 * per-pixel step and blending (raster_alpha_2dgs.cuh) on those only. A skipped entry has
 * alpha below the threshold at every pixel of the subtile, which the default
 * kernel skips per pixel, so the outputs are identical; this kernel only
 * decides which entries each warp visits.
 ****************************************************************************/

namespace subtile_fwd_2dgs {
using namespace subtile_layout;

// Corner coordinates of the subtile boxes, relative to the tile: the box of
// subtile (column c, row r) spans grid_x(c)..grid_x(c + 1) and
// grid_y(r)..grid_y(r + 1), which covers its pixel centres; neighbouring boxes
// share corners, so the ray-splat map is evaluated at (SUB_COLS + 1) x
// (TILE / SUB_H + 1) points. The outer corners are the outermost pixel centres.
inline __device__ float grid_x(const uint32_t c) {
    return c == 0 ? 0.5f : (c == SUB_COLS ? TILE - 0.5f : (float)(c * SUB_W));
}
inline __device__ float grid_y(const uint32_t r) {
    return r == 0 ? 0.5f : (r == TILE / SUB_H ? TILE - 0.5f : (float)(r * SUB_H));
}

// A box corner mapped into the splat's (u, v) frame, with a bound on the
// rounding error of the kernel's per-pixel evaluation near it.
template <typename S> struct UvCorner {
    S u, v;   // intersection in the splat frame
    S z;      // ray_cross.z (its sign must not change across a box)
    S err;    // rounding scale of the ray_cross components (absolute)
};

template <typename S>
inline __device__ UvCorner<S> uv_corner(
    const vec3<S> u_M, const vec3<S> v_M, const vec3<S> w_M, const S px, const S py
) {
    // the per-pixel formula of surfel_alpha
    const vec3<S> h_u = px * w_M - u_M;
    const vec3<S> h_v = py * w_M - v_M;
    const vec3<S> rc = glm::cross(h_u, h_v);
    // each ray_cross component is a difference of products of h components,
    // which are themselves rounded: a few ulps of |h_u| |h_v|, and of the
    // magnitude of the products forming h
    const S hu = fabsf(px) * (fabsf(w_M.x) + fabsf(w_M.y) + fabsf(w_M.z)) +
                 fabsf(u_M.x) + fabsf(u_M.y) + fabsf(u_M.z);
    const S hv = fabsf(py) * (fabsf(w_M.x) + fabsf(w_M.y) + fabsf(w_M.z)) +
                 fabsf(v_M.x) + fabsf(v_M.y) + fabsf(v_M.z);
    UvCorner<S> c;
    c.z = rc.z;
    c.err = 16.f * 5.9604645e-8f * hu * hv; // 16 ulps of the product bound
    c.u = rc.x / rc.z;
    c.v = rc.y / rc.z;
    return c;
}

// Does the convex quad a-b-c-d (in order) come within distance r of the origin?
template <typename S>
inline __device__ bool quad_near_origin(
    const UvCorner<S> &a, const UvCorner<S> &b, const UvCorner<S> &c,
    const UvCorner<S> &d, const S r
) {
    const S px[4] = {a.u, b.u, c.u, d.u};
    const S py[4] = {a.v, b.v, c.v, d.v};
    bool pos = true, neg = true;
    const S r2 = r * r;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const S x0 = px[k], y0 = py[k];
        const S ex = px[(k + 1) % 4] - x0, ey = py[(k + 1) % 4] - y0;
        // origin side of the edge
        const S cr = ex * (-y0) - ey * (-x0);
        pos = pos && cr >= 0.f;
        neg = neg && cr <= 0.f;
        // distance from the origin to the edge segment
        const S len2 = ex * ex + ey * ey;
        S t = len2 > 0.f ? -(x0 * ex + y0 * ey) / len2 : 0.f;
        t = fminf(fmaxf(t, 0.f), 1.f);
        const S qx = x0 + t * ex, qy = y0 + t * ey;
        if (qx * qx + qy * qy <= r2) {
            return true;
        }
    }
    return pos || neg; // the origin lies inside the quad
}

// Subtiles where surfel_alpha may accept the surfel at some pixel centre. The
// weight is the smaller of the ray-splat |s|^2 and the screen-space low-pass
// FILTER_INV_SQUARE |p - mean2d|^2, so a subtile is reached if either is at
// most 2 tau, tau = ln(ALPHA_THRESHOLD_INV * opacity):
//   - low-pass: the subtile's pixel centres come within sqrt(2 tau /
//     FILTER_INV_SQUARE) of mean2d;
//   - ray-splat: the subtile box, mapped into the splat's (u, v) frame, comes
//     within sqrt(2 tau) of the splat centre. While ray_cross.z keeps its sign
//     across the box the map is projective and the image a convex quad with
//     the mapped corners as vertices; otherwise (or when the sign is not
//     reliable in float32) the subtile counts as reached.
// Margins cover __expf/__logf and float32 rounding of the per-pixel
// evaluation. It never clears a bit the per-pixel test would accept; it may set
// extra ones. Non-finite input reaches every subtile (checked before the
// opacity, as a NaN weight passes surfel_alpha at MAX_ALPHA).
template <typename S>
inline __device__ uint32_t subtile_mask_2dgs(
    const vec3<S> xy_opac,
    const vec3<S> u_M,
    const vec3<S> v_M,
    const vec3<S> w_M,
    const S tile_x0,
    const S tile_y0
) {
    if (!isfinite(xy_opac.x) || !isfinite(xy_opac.y) || !isfinite(u_M.x) ||
        !isfinite(u_M.y) || !isfinite(u_M.z) || !isfinite(v_M.x) ||
        !isfinite(v_M.y) || !isfinite(v_M.z) || !isfinite(w_M.x) ||
        !isfinite(w_M.y) || !isfinite(w_M.z)) {
        return 0xffu;
    }
    const S opac = xy_opac.z;
    if (!(opac * ALPHA_THRESHOLD_INV >= 1.f)) {
        return isfinite(opac) ? 0u : 0xffu;
    }
    const S tau = __logf(opac * ALPHA_THRESHOLD_INV) * 1.01f + 1e-3f;
    if (!isfinite(tau)) {
        return 0xffu;
    }

    uint32_t mask = 0u;
    // low-pass part
    const S r2d_sq = 2.f * tau / FILTER_INV_SQUARE * 1.001f + 1e-3f;
#pragma unroll
    for (uint32_t w = 0; w < SUBTILES; ++w) {
        const S x0 = tile_x0 + (S)subtile_x0(w) + 0.5f;
        const S y0 = tile_y0 + (S)subtile_y0(w) + 0.5f;
        const S dx = fmaxf(fmaxf(x0 - xy_opac.x, xy_opac.x - (x0 + (S)SUB_W - 1.f)), 0.f);
        const S dy = fmaxf(fmaxf(y0 - xy_opac.y, xy_opac.y - (y0 + (S)SUB_H - 1.f)), 0.f);
        mask |= (dx * dx + dy * dy <= r2d_sq) ? (1u << w) : 0u;
    }

    // ray-splat part, row by row over the 3 x 5 corner grid
    const S r3d = sqrtf(2.f * tau) * 1.01f + 1e-3f;
    UvCorner<S> prev[SUB_COLS + 1], cur[SUB_COLS + 1];
#pragma unroll
    for (uint32_t row = 0; row <= TILE / SUB_H; ++row) {
#pragma unroll
        for (uint32_t col = 0; col <= SUB_COLS; ++col) {
            cur[col] = uv_corner<S>(
                u_M, v_M, w_M, tile_x0 + (S)grid_x(col), tile_y0 + (S)grid_y(row)
            );
        }
        if (row > 0) {
#pragma unroll
            for (uint32_t col = 0; col < SUB_COLS; ++col) {
                const uint32_t w = (row - 1) * SUB_COLS + col;
                if (mask & (1u << w)) {
                    continue;
                }
                const UvCorner<S> &a = prev[col], &b = prev[col + 1];
                const UvCorner<S> &c = cur[col + 1], &d = cur[col];
                // ray_cross.z is linear in the pixel, so over the box it lies
                // between its corner values; demand one sign with a margin
                const S zmin = fminf(fminf(a.z, b.z), fminf(c.z, d.z));
                const S zmax = fmaxf(fmaxf(a.z, b.z), fmaxf(c.z, d.z));
                const S err = fmaxf(fmaxf(a.err, b.err), fmaxf(c.err, d.err));
                const S zabs = fminf(fabsf(zmin), fabsf(zmax));
                const bool one_sign = (zmin > 0.f || zmax < 0.f) && zabs > 16.f * err;
                bool reach = !one_sign;
                if (!reach) {
                    // error of u, v in the box: (err + |s| err) / |z|
                    const S smax = fmaxf(
                        fmaxf(fabsf(a.u) + fabsf(a.v), fabsf(b.u) + fabsf(b.v)),
                        fmaxf(fabsf(c.u) + fabsf(c.v), fabsf(d.u) + fabsf(d.v))
                    );
                    const S r = r3d + 2.f * err * (1.f + smax) / zabs;
                    reach = !isfinite(r) || quad_near_origin<S>(a, b, c, d, r);
                }
                mask |= reach ? (1u << w) : 0u;
            }
        }
#pragma unroll
        for (uint32_t col = 0; col <= SUB_COLS; ++col) {
            prev[col] = cur[col];
        }
    }
    return mask;
}

// Unlike the 3DGS subtile kernel, colours are not staged in shared memory:
// with the three ray-transform rows per entry the batch already takes 17 KB,
// and staging 1-16 channels was slower or equal on every measured case (water2
// surfels, 1280x720 and 3840x2160).
} // namespace subtile_fwd_2dgs

template <uint32_t COLOR_DIM, typename S>
__global__ void __launch_bounds__(subtile_fwd_2dgs::THREADS)
    rasterize_to_pixels_fwd_2dgs_subtile_kernel(
        const uint32_t C,
        const uint32_t n_isects,
        const vec2<S> *__restrict__ means2d,
        const S *__restrict__ ray_transforms,
        const S *__restrict__ colors,
        const S *__restrict__ opacities,
        const S *__restrict__ normals,
        const S *__restrict__ backgrounds,
        const bool *__restrict__ masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ flatten_ids,
        S *__restrict__ render_colors,
        S *__restrict__ render_alphas,
        S *__restrict__ render_normals,
        S *__restrict__ render_distort,
        S *__restrict__ render_median,
        int32_t *__restrict__ last_ids,
        int32_t *__restrict__ median_ids
    ) {
    using namespace subtile_fwd_2dgs;
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t camera_id = block.group_index().x;
    const uint32_t tile_row = block.group_index().y;
    const uint32_t tile_col = block.group_index().z;
    const int32_t tile_id = tile_row * tile_width + tile_col;
    const uint32_t tr = block.thread_rank();
    const uint32_t warp_id = tr / 32;
    const uint32_t lane = warp.thread_rank();
    // this thread's pixel: lane of the warp's subtile, row-major
    const uint32_t i = tile_row * TILE + subtile_y0(warp_id) + lane / SUB_W;
    const uint32_t j = tile_col * TILE + subtile_x0(warp_id) + lane % SUB_W;

    const int64_t pixels = (int64_t)image_height * image_width;
    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * pixels * COLOR_DIM;
    render_alphas += camera_id * pixels;
    render_normals += camera_id * pixels * 3;
    if (render_distort != nullptr) {
        render_distort += camera_id * pixels;
    }
    render_median += camera_id * pixels;
    last_ids += camera_id * pixels;
    median_ids += camera_id * pixels;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // a masked tile writes empty pixels; the whole block leaves together, so
    // no warp reaches the ballots below with some lanes exited
    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
            write_empty_surfel_pixel<COLOR_DIM, S>(
                pix_id, backgrounds, render_colors, render_alphas, render_normals,
                render_distort, render_median, last_ids, median_ids
            );
        }
        return;
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches = (range_end - range_start + THREADS - 1) / THREADS;

    __shared__ int32_t id_batch[THREADS];
    __shared__ vec3<S> xy_opacity_batch[THREADS];
    __shared__ vec3<S> u_Ms_batch[THREADS];
    __shared__ vec3<S> v_Ms_batch[THREADS];
    __shared__ vec3<S> w_Ms_batch[THREADS];
    __shared__ uint8_t mask_batch[THREADS];
    __shared__ uint16_t hits[SUBTILES][THREADS]; // per-warp hit list of entries

    const S tile_x0 = (S)(tile_col * TILE);
    const S tile_y0 = (S)(tile_row * TILE);
    S T = 1.0f;
    uint32_t cur_idx = 0;
    S pix_out[COLOR_DIM] = {0.f};
    S normal_out[3] = {0.f};
    S distort = 0.f;
    S accum_vis_depth = 0.f;
    S median_depth = 0.f;
    uint32_t median_idx = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= THREADS) {
            break;
        }
        const uint32_t batch_start = range_start + THREADS * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const vec3<S> xy_opac = {xy.x, xy.y, opacities[g]};
            const S *M = ray_transforms + g * 9;
            const vec3<S> u_M = {M[0], M[1], M[2]};
            const vec3<S> v_M = {M[3], M[4], M[5]};
            const vec3<S> w_M = {M[6], M[7], M[8]};
            xy_opacity_batch[tr] = xy_opac;
            u_Ms_batch[tr] = u_M;
            v_Ms_batch[tr] = v_M;
            w_Ms_batch[tr] = w_M;
            mask_batch[tr] = (uint8_t)subtile_mask_2dgs<S>(
                xy_opac, u_M, v_M, w_M, tile_x0, tile_y0
            );
        }
        block.sync();

        // compact this warp's entries, keeping depth order
        const uint32_t batch_size = min(THREADS, range_end - batch_start);
        const uint32_t n_hits =
            compact_warp_hits(warp, warp_id, mask_batch, batch_size, hits[warp_id]);
        warp.sync();

        for (uint32_t h = 0; (h < n_hits) && !done; ++h) {
            const uint32_t t = hits[warp_id][h];
            // the default kernel's per-pixel step (raster_alpha_2dgs.cuh)
            S alpha;
            if (!surfel_alpha<S>(
                    xy_opacity_batch[t], u_Ms_batch[t], v_Ms_batch[t],
                    w_Ms_batch[t], px, py, alpha
                )) {
                continue;
            }
            const S next_T = surfel_next_T(T, alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }
            const int32_t g = id_batch[t];
            surfel_blend<COLOR_DIM, S>(
                alpha, T, colors + g * COLOR_DIM, normals + g * 3, pix_out,
                normal_out, render_distort != nullptr, distort, accum_vis_depth,
                median_depth, median_idx, batch_start + t
            );
            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : __fmaf_rn(T, backgrounds[k], pix_out[k]);
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        if (render_distort != nullptr) {
            render_distort[pix_id] = distort;
        }
        render_median[pix_id] = median_depth;
        median_ids[pix_id] = static_cast<int32_t>(median_idx);
    }
}

// 2DGS subtile forward into the preallocated outputs (tile size 16).
template <uint32_t CDIM>
void launch_rasterize_to_pixels_fwd_2dgs_subtile(
    const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
    const torch::Tensor &colors,         // [C, N, CDIM] or [nnz, CDIM]
    const torch::Tensor &opacities,      // [C, N] or [nnz]
    const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, CDIM]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    torch::Tensor &renders,
    torch::Tensor &alphas,
    torch::Tensor &render_normals,
    torch::Tensor &render_distort,
    torch::Tensor &render_median,
    torch::Tensor &last_ids,
    torch::Tensor &median_ids,
    const cudaStream_t stream
) {
    TORCH_CHECK(
        tile_size == subtile_fwd_2dgs::TILE,
        "the subtile forward needs tile_size 16, got ",
        tile_size
    );
    const uint32_t C = tile_offsets.size(0);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const dim3 blocks = {C, tile_height, tile_width};
    rasterize_to_pixels_fwd_2dgs_subtile_kernel<CDIM, float>
        <<<blocks, subtile_fwd_2dgs::THREADS, 0, stream>>>(
            C,
            flatten_ids.size(0),
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_distort.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>()
        );
}

} // namespace gsplat

#endif // GSPLAT_RASTERIZE_TO_PIXELS_2DGS_FWD_SUBTILE_CUH
