#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include "warp_reduce_scatter.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

// LEAN (PennyGS, arXiv 2611.27525 Sec. 3.6 "less state per pixel"): the colors
// behind a Gaussian enter dL/dalpha only through their dot product with the
// pixel's color gradient, so a pixel keeps that scalar instead of COLOR_DIM
// partial sums. Same math; the summation order of that term changes.
template <uint32_t COLOR_DIM, bool LEAN, typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                           // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    // Tile-uniform exit, before any collectives or exclusive-end arithmetic.
    if (range_start == range_end) {
        return;
    }
    const uint32_t block_size = block.size();

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        );                                         // [block_size]
    S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one: per channel, or
    // (LEAN) only its dot product with this pixel's color gradient
    S buffer[LEAN ? 1 : COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    // last_ids are absolute intersection positions. Trim the tile's fully
    // occluded suffix before staging backwards, not only inside each warp's
    // pixel loop. Keep the original positions/IDs and do not mutate binning
    // buffers, which may be reused by other channel chunks.
    if (tr == 0) {
        id_batch[0] = range_start;
    }
    block.sync();
    if (warp.thread_rank() == 0) {
        atomicMax(id_batch, warp_bin_final);
    }
    block.sync();
    const int64_t exclusive_end = static_cast<int64_t>(id_batch[0]) + 1;
    range_end = static_cast<int32_t>(
        exclusive_end < range_end ? exclusive_end : range_end
    );
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            vec2<S> delta;
            vec3<S> conic;
            S vis;

            if (valid || COLOR_DIM <= 4) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                  conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
                // Narrow-channel pixels stay converged through the recurrence.
                // Misses use the identity transmittance update. Final gradient
                // contributions and buffer updates must still be gated: zero
                // factors alone do not suppress nonfinite operands.
                alpha = valid ? alpha : 0.f;
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid || COLOR_DIM <= 4) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = valid ? fac * v_render_c[k] : 0.f;
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                S rgb_dot_v = 0.f; // LEAN: this Gaussian's color . dL/dcolor
                if constexpr (LEAN) {
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        rgb_dot_v += rgbs_batch[t * COLOR_DIM + k] * v_render_c[k];
                    }
                    v_alpha = rgb_dot_v * T - buffer[0] * ra;
                } else {
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T -
                                    buffer[k] * ra) *
                                   v_render_c[k];
                    }
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                const bool differentiable = valid && opac * vis <= 0.999f;
                // Keep initialized zeros when invalid or clamped, rather than
                // multiplying zero by potentially overflowing geometry.
                if (differentiable) {
                    const S v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

                if (valid) {
                    if constexpr (LEAN) {
                        buffer[0] += rgb_dot_v * fac;
                    } else {
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                        }
                    }
                }
            }
            auto standard_reduce = [&]() {
                warpSum<COLOR_DIM, S>(v_rgb_local, warp);
                warpSum<decltype(warp), S>(v_conic_local, warp);
                warpSum<decltype(warp), S>(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum<decltype(warp), S>(v_xy_abs_local, warp);
                }
                warpSum<decltype(warp), S>(v_opacity_local, warp);
                if (warp.thread_rank() == 0) {
                    int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                    S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }

                    S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                    S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                    if (v_means2d_abs != nullptr) {
                        S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }

                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                }
            };
            if constexpr (COLOR_DIM <= 8) {
                // Scatter requires all 32 lanes, including upper component
                // owners. Use the baseline path for any partial-warp block.
                if (block_size % 32 != 0) {
                    standard_reduce();
                    continue;
                }
                // RGB/depth have a narrow gradient vector: distribute its
                // reduced components across lanes instead of reducing every
                // component redundantly and serializing all atomics in lane 0.
                // Packed gradient layout. Reuse these boundaries for both
                // packing and writeback so component offsets have one owner.
                constexpr int conic_begin = COLOR_DIM;
                constexpr int mean_begin = conic_begin + 3;
                constexpr int opacity_index = mean_begin + 2;
                constexpr int abs_mean_begin = opacity_index + 1;
                constexpr int gradient_count = abs_mean_begin + 2;
                S values[gradient_count];
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    values[k] = v_rgb_local[k];
                }
                values[conic_begin] = v_conic_local.x;
                values[conic_begin + 1] = v_conic_local.y;
                values[conic_begin + 2] = v_conic_local.z;
                values[mean_begin] = v_xy_local.x;
                values[mean_begin + 1] = v_xy_local.y;
                values[opacity_index] = v_opacity_local;
                values[abs_mean_begin] = v_xy_abs_local.x;
                values[abs_mean_begin + 1] = v_xy_abs_local.y;
                const auto reduced = warp_reduce_scatter(values, warp);
                if (reduced.owner) {
                    const int component = reduced.component;
                    int32_t g = id_batch[t];
                    S *destination = nullptr;
                    if (component < conic_begin) {
                        destination = v_colors + COLOR_DIM * g + component;
                    } else if (component < mean_begin) {
                        destination =
                            (S *)v_conics + 3 * g + component - conic_begin;
                    } else if (component < opacity_index) {
                        destination =
                            (S *)v_means2d + 2 * g + component - mean_begin;
                    } else if (component == opacity_index) {
                        destination = v_opacities + g;
                    } else if (v_means2d_abs != nullptr) {
                        destination =
                            (S *)v_means2d_abs + 2 * g + component - abs_mean_begin;
                    }
                    if (destination != nullptr) {
                        gpuAtomicAdd(destination, reduced.value);
                    }
                }
            } else {
                standard_reduce();
            }
        }
    }
}

/****************************************************************************
 * Per-chunk backward traversal (PennyGS, arXiv 2611.27525 Sec. 3.6)
 *
 * A 16x16 tile is eight 8x4 pixel chunks. Four warps share the tile; a warp
 * owns one Gaussian at a time and walks the chunks, its 32 lanes handling the
 * chunk's 32 pixels. Pixel state (transmittance, the partial sums of the
 * Gaussians behind, the color gradient) lives in shared memory. A pixel must
 * meet its Gaussians back to front, so a Gaussian enters a chunk only after
 * every earlier (further back) Gaussian that walks the chunk has left it;
 * per-chunk counters order the warps. The warp reduces a Gaussian's gradient
 * over all its pixels once and issues one set of atomics per tile.
 *
 * CULL ("chunk culling"): a Gaussian walks, and waits for, only the chunks its
 * alpha >= 1/255 ellipse can reach (conservative bounding-box test).
 * LEAN: as in the per-pixel kernel, keep only the dot product of the behind
 * colors with the pixel's color gradient.
 ****************************************************************************/

namespace chunk_bwd {
constexpr uint32_t TILE = 16;
constexpr uint32_t CHUNKS = 8; // 8x4 pixels each: 2 columns x 4 rows
constexpr uint32_t WARPS = 4;
constexpr uint32_t THREADS = WARPS * 32;
constexpr uint32_t BATCH = THREADS; // Gaussians staged per batch

inline __device__ uint32_t pixel_of(uint32_t chunk, uint32_t lane) {
    const uint32_t x = (chunk % 2) * 8 + lane % 8;
    const uint32_t y = (chunk / 2) * 4 + lane / 8;
    return y * TILE + x;
}

// Chunks whose pixel centres can see alpha >= 1/255 from this Gaussian. Bounding
// box of {d : 0.5 d^T Q d <= ln(255 * opacity)} with a margin; any degenerate or
// nonfinite input reaches every chunk, so culling never drops a valid pixel.
template <typename S>
inline __device__ uint32_t chunk_mask(
    const vec2<S> xy, const vec3<S> conic, const S opac, const S tile_x0,
    const S tile_y0
) {
    if (!(opac * 255.f >= 1.f)) {
        return isfinite(opac) ? 0u : 0xffu;
    }
    const S tau = __logf(opac * 255.f) * 1.01f + 1e-3f;
    const S det = conic.x * conic.z - conic.y * conic.y;
    if (!(det > 0.f) || !isfinite(det) || !isfinite(xy.x) || !isfinite(xy.y)) {
        return 0xffu;
    }
    const S half_x = sqrtf(2.f * tau * conic.z / det) + 0.05f;
    const S half_y = sqrtf(2.f * tau * conic.x / det) + 0.05f;
    if (!isfinite(half_x) || !isfinite(half_y)) {
        return 0xffu;
    }
    uint32_t mask = 0u;
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        // pixel centres of the chunk span [x0 + 0.5, x0 + 7.5] x [y0 + 0.5, y0 + 3.5]
        const S x0 = tile_x0 + (c % 2) * 8.f;
        const S y0 = tile_y0 + (c / 2) * 4.f;
        const bool reach = xy.x + half_x >= x0 + 0.5f && xy.x - half_x <= x0 + 7.5f &&
                           xy.y + half_y >= y0 + 0.5f && xy.y - half_y <= y0 + 3.5f;
        mask |= reach ? (1u << c) : 0u;
    }
    return mask;
}
} // namespace chunk_bwd

template <uint32_t COLOR_DIM, bool LEAN, bool CULL, typename S>
__global__ void __launch_bounds__(chunk_bwd::THREADS)
    rasterize_to_pixels_bwd_chunk_kernel(
        const uint32_t C,
        const uint32_t n_isects,
        const vec2<S> *__restrict__ means2d,
        const vec3<S> *__restrict__ conics,
        const S *__restrict__ colors,
        const S *__restrict__ opacities,
        const S *__restrict__ backgrounds,
        const bool *__restrict__ masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ flatten_ids,
        const S *__restrict__ render_alphas,
        const int32_t *__restrict__ last_ids,
        const S *__restrict__ v_render_colors,
        const S *__restrict__ v_render_alphas,
        vec2<S> *__restrict__ v_means2d_abs,
        vec2<S> *__restrict__ v_means2d,
        vec3<S> *__restrict__ v_conics,
        S *__restrict__ v_colors,
        S *__restrict__ v_opacities
    ) {
    using namespace chunk_bwd;
    static_assert(COLOR_DIM <= 8, "chunk backward supports up to 8 channels");
    constexpr uint32_t PIXELS = TILE * TILE;
    constexpr uint32_t BUF = LEAN ? 1 : COLOR_DIM;

    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t camera_id = block.group_index().x;
    const uint32_t tile_row = block.group_index().y;
    const uint32_t tile_col = block.group_index().z;
    const uint32_t tile_id = tile_row * tile_width + tile_col;
    const uint32_t tr = block.thread_rank();
    const uint32_t warp_id = tr / 32;
    const uint32_t lane = warp.thread_rank();

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
        if (!masks[tile_id]) {
            return;
        }
    }
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    if (range_start == range_end) {
        return;
    }

    // Pixel state.
    __shared__ S s_T[PIXELS];
    __shared__ S s_buf[PIXELS * BUF];
    __shared__ S s_vc[PIXELS * COLOR_DIM];
    __shared__ S s_wfinal[PIXELS]; // T_final * (dL/dalpha_out - bg . dL/dcolor)
    __shared__ int32_t s_bin[PIXELS];
    // Gaussian batch.
    __shared__ int32_t s_id[BATCH];
    __shared__ vec3<S> s_xyo[BATCH];
    __shared__ vec3<S> s_conic[BATCH];
    __shared__ S s_rgb[BATCH * COLOR_DIM];
    __shared__ uint8_t s_mask[BATCH];
    __shared__ uint8_t s_rank[CHUNKS * BATCH];
    __shared__ uint32_t s_warp_count[CHUNKS * WARPS];
    __shared__ int32_t s_done[CHUNKS];
    __shared__ int32_t s_last;

    if (tr == 0) {
        s_last = range_start;
    }
    block.sync();
    const S tile_x0 = (S)(tile_col * TILE);
    const S tile_y0 = (S)(tile_row * TILE);
    for (uint32_t p = tr; p < PIXELS; p += THREADS) {
        const uint32_t i = tile_row * TILE + p / TILE;
        const uint32_t j = tile_col * TILE + p % TILE;
        const bool inside = i < image_height && j < image_width;
        const uint32_t pix = inside ? i * image_width + j : 0;
        const S T_final = inside ? 1.0f - render_alphas[pix] : 1.0f;
        s_T[p] = T_final;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < BUF; ++k) {
            s_buf[p * BUF + k] = 0.f;
        }
        S bg_dot_v = 0.f;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            const S v = inside ? v_render_colors[pix * COLOR_DIM + k] : 0.f;
            s_vc[p * COLOR_DIM + k] = v;
            if (backgrounds != nullptr) {
                bg_dot_v += backgrounds[k] * v;
            }
        }
        const S v_a = inside ? v_render_alphas[pix] : 0.f;
        s_wfinal[p] = T_final * (v_a - bg_dot_v);
        const int32_t bin = inside ? last_ids[pix] : -1;
        s_bin[p] = bin;
        if (inside) {
            atomicMax(&s_last, bin);
        }
    }
    block.sync();
    // Trim the tile's fully occluded suffix, as the per-pixel kernel does.
    const int64_t exclusive_end = static_cast<int64_t>(s_last) + 1;
    range_end = static_cast<int32_t>(
        exclusive_end < range_end ? exclusive_end : range_end
    );

    const uint32_t num_batches = (range_end - range_start + BATCH - 1) / BATCH;
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        // order k = 0 is the furthest back Gaussian of the batch
        const int32_t batch_end = range_end - 1 - BATCH * b;
        const int32_t batch_size = min((int32_t)BATCH, batch_end + 1 - range_start);
        uint32_t mask = 0u;
        if ((int32_t)tr < batch_size) {
            const int32_t g = flatten_ids[batch_end - tr];
            s_id[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            const vec3<S> conic = conics[g];
            s_xyo[tr] = {xy.x, xy.y, opac};
            s_conic[tr] = conic;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                s_rgb[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
            mask = CULL ? chunk_mask<S>(xy, conic, opac, tile_x0, tile_y0) : 0xffu;
            s_mask[tr] = (uint8_t)mask;
        }
        if constexpr (CULL) {
            // rank[c][k]: number of earlier Gaussians of the batch walking chunk c
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t c = 0; c < CHUNKS; ++c) {
                const uint32_t ballot = warp.ballot((mask >> c) & 1u);
                s_rank[c * BATCH + tr] = __popc(ballot & ((1u << lane) - 1u));
                if (lane == 0) {
                    s_warp_count[c * WARPS + warp_id] = __popc(ballot);
                }
            }
            block.sync();
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t c = 0; c < CHUNKS; ++c) {
                uint32_t base = 0;
                for (uint32_t w = 0; w < warp_id; ++w) {
                    base += s_warp_count[c * WARPS + w];
                }
                s_rank[c * BATCH + tr] += base;
            }
        }
        if (tr < CHUNKS) {
            s_done[tr] = 0;
        }
        block.sync();

        for (int32_t k = warp_id; k < batch_size; k += WARPS) {
            const int32_t idx = batch_end - k; // position in the tile's list
            const vec3<S> xyo = s_xyo[k];
            const vec3<S> conic = s_conic[k];
            const S opac = xyo.z;
            S rgb[COLOR_DIM];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t c = 0; c < COLOR_DIM; ++c) {
                rgb[c] = s_rgb[k * COLOR_DIM + c];
            }
            const uint32_t walk = CULL ? s_mask[k] : 0xffu;

            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;

            for (uint32_t c = 0; c < CHUNKS; ++c) {
                if (!((walk >> c) & 1u)) {
                    continue;
                }
                const int32_t rank = CULL ? (int32_t)s_rank[c * BATCH + k] : k;
                // Wait until every earlier Gaussian walking this chunk has left
                // (acquire: their pixel-state writes are visible after the warp
                // barrier).
                cuda::atomic_ref<int32_t, cuda::thread_scope_block> done(s_done[c]);
                if (lane == 0) {
                    while (done.load(cuda::memory_order_acquire) != rank) {
                    }
                }
                warp.sync();

                const uint32_t p = pixel_of(c, lane);
                const S px = tile_x0 + (S)(p % TILE) + 0.5f;
                const S py = tile_y0 + (S)(p / TILE) + 0.5f;
                bool valid = idx <= s_bin[p];
                S alpha = 0.f, vis = 0.f;
                vec2<S> delta = {0.f, 0.f};
                if (valid) {
                    delta = {xyo.x - px, xyo.y - py};
                    const S sigma =
                        0.5f * (conic.x * delta.x * delta.x +
                                conic.z * delta.y * delta.y) +
                        conic.y * delta.x * delta.y;
                    vis = __expf(-sigma);
                    alpha = min(0.999f, opac * vis);
                    if (sigma < 0.f || alpha < 1.f / 255.f) {
                        valid = false;
                    }
                }
                if (valid) {
                    const S ra = 1.0f / (1.0f - alpha);
                    const S T = s_T[p] * ra;
                    s_T[p] = T;
                    const S fac = alpha * T;
                    S v_alpha = 0.f;
                    S rgb_dot_v = 0.f;
                    if constexpr (LEAN) {
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t q = 0; q < COLOR_DIM; ++q) {
                            const S v = s_vc[p * COLOR_DIM + q];
                            rgb_dot_v += rgb[q] * v;
                            v_rgb_local[q] += fac * v;
                        }
                        v_alpha = rgb_dot_v * T - s_buf[p] * ra;
                        s_buf[p] += rgb_dot_v * fac;
                    } else {
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t q = 0; q < COLOR_DIM; ++q) {
                            const S v = s_vc[p * COLOR_DIM + q];
                            v_rgb_local[q] += fac * v;
                            v_alpha += (rgb[q] * T - s_buf[p * COLOR_DIM + q] * ra) * v;
                            s_buf[p * COLOR_DIM + q] += rgb[q] * fac;
                        }
                    }
                    v_alpha += s_wfinal[p] * ra;
                    if (opac * vis <= 0.999f) {
                        const S v_sigma = -opac * vis * v_alpha;
                        v_conic_local.x += 0.5f * v_sigma * delta.x * delta.x;
                        v_conic_local.y += v_sigma * delta.x * delta.y;
                        v_conic_local.z += 0.5f * v_sigma * delta.y * delta.y;
                        const vec2<S> v_xy = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        v_xy_local.x += v_xy.x;
                        v_xy_local.y += v_xy.y;
                        v_xy_abs_local.x += abs(v_xy.x);
                        v_xy_abs_local.y += abs(v_xy.y);
                        v_opacity_local += vis * v_alpha;
                    }
                }
                // Publish this chunk's pixel state to the next Gaussian: the
                // warp barrier orders every lane's writes before the release.
                warp.sync();
                if (lane == 0) {
                    done.store(rank + 1, cuda::memory_order_release);
                }
            }

            // One reduction and one set of atomics per Gaussian and tile.
            constexpr int conic_begin = COLOR_DIM;
            constexpr int mean_begin = conic_begin + 3;
            constexpr int opacity_index = mean_begin + 2;
            constexpr int abs_mean_begin = opacity_index + 1;
            constexpr int gradient_count = abs_mean_begin + 2;
            S values[gradient_count];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t q = 0; q < COLOR_DIM; ++q) {
                values[q] = v_rgb_local[q];
            }
            values[conic_begin] = v_conic_local.x;
            values[conic_begin + 1] = v_conic_local.y;
            values[conic_begin + 2] = v_conic_local.z;
            values[mean_begin] = v_xy_local.x;
            values[mean_begin + 1] = v_xy_local.y;
            values[opacity_index] = v_opacity_local;
            values[abs_mean_begin] = v_xy_abs_local.x;
            values[abs_mean_begin + 1] = v_xy_abs_local.y;
            const auto reduced = warp_reduce_scatter(values, warp);
            if (reduced.owner) {
                const int component = reduced.component;
                const int32_t g = s_id[k];
                S *destination = nullptr;
                if (component < conic_begin) {
                    destination = v_colors + COLOR_DIM * g + component;
                } else if (component < mean_begin) {
                    destination = (S *)v_conics + 3 * g + component - conic_begin;
                } else if (component < opacity_index) {
                    destination = (S *)v_means2d + 2 * g + component - mean_begin;
                } else if (component == opacity_index) {
                    destination = v_opacities + g;
                } else if (v_means2d_abs != nullptr) {
                    destination =
                        (S *)v_means2d_abs + 2 * g + component - abs_mean_begin;
                }
                if (destination != nullptr) {
                    gpuAtomicAdd(destination, reduced.value);
                }
            }
        }
    }
}

// Backward traversal choices (`rasterize_to_pixels(..., backward=...)`).
enum class BackwardKernel : int64_t {
    PIXEL = 0,           // per-pixel threads (default)
    PIXEL_LEAN = 1,      // per-pixel threads, scalar behind-color state
    CHUNK = 2,           // per-chunk warps
    CHUNK_LEAN = 3,      // per-chunk warps, scalar behind-color state
    CHUNK_CULL = 4,      // per-chunk warps, chunk culling
    CHUNK_LEAN_CULL = 5, // per-chunk warps, scalar state and chunk culling
};

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
    BackwardKernel backward_kernel
) {

    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    const bool chunk = backward_kernel != BackwardKernel::PIXEL &&
                       backward_kernel != BackwardKernel::PIXEL_LEAN;
    if (n_isects && chunk) {
        if constexpr (CDIM <= 8) {
            TORCH_CHECK(
                tile_size == chunk_bwd::TILE,
                "chunk backward needs tile_size 16, got ",
                tile_size
            );
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
#define __GS__CHUNK_(LEAN, CULL)                                                \
    rasterize_to_pixels_bwd_chunk_kernel<CDIM, LEAN, CULL, float>              \
        <<<blocks, chunk_bwd::THREADS, 0, stream>>>(                           \
            C,                                                                 \
            n_isects,                                                          \
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),       \
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),        \
            colors.data_ptr<float>(),                                          \
            opacities.data_ptr<float>(),                                       \
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()    \
                                    : nullptr,                                 \
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,      \
            image_width,                                                       \
            image_height,                                                      \
            tile_width,                                                        \
            tile_height,                                                       \
            tile_offsets.data_ptr<int32_t>(),                                  \
            flatten_ids.data_ptr<int32_t>(),                                   \
            render_alphas.data_ptr<float>(),                                   \
            last_ids.data_ptr<int32_t>(),                                      \
            v_render_colors.data_ptr<float>(),                                 \
            v_render_alphas.data_ptr<float>(),                                 \
            absgrad ? reinterpret_cast<vec2<float> *>(                         \
                          v_means2d_abs.data_ptr<float>()                      \
                      )                                                        \
                    : nullptr,                                                 \
            reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),     \
            reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),      \
            v_colors.data_ptr<float>(),                                        \
            v_opacities.data_ptr<float>()                                      \
        );
            switch (backward_kernel) {
            case BackwardKernel::CHUNK:
                __GS__CHUNK_(false, false) break;
            case BackwardKernel::CHUNK_LEAN:
                __GS__CHUNK_(true, false) break;
            case BackwardKernel::CHUNK_CULL:
                __GS__CHUNK_(false, true) break;
            default:
                __GS__CHUNK_(true, true) break;
            }
#undef __GS__CHUNK_
        } else {
            AT_ERROR("chunk backward supports up to 8 channels, got ", CDIM);
        }
    } else if (n_isects) {
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        const bool lean = backward_kernel == BackwardKernel::PIXEL_LEAN;
        const auto kernel = lean ? rasterize_to_pixels_bwd_kernel<CDIM, true, float>
                                 : rasterize_to_pixels_bwd_kernel<CDIM, false, float>;

        if (cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }
        kernel<<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? reinterpret_cast<vec2<float> *>(
                              v_means2d_abs.data_ptr<float>()
                          )
                        : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>()
            );
    }

    return std::make_tuple(
        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad,
    int64_t backward_kernel
) {

    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);
    TORCH_CHECK(
        backward_kernel >= 0 && backward_kernel <= 5,
        "Unknown backward kernel ",
        backward_kernel
    );

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_alphas,                                                     \
            last_ids,                                                          \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            absgrad,                                                           \
            static_cast<BackwardKernel>(backward_kernel)                       \
        );

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat
