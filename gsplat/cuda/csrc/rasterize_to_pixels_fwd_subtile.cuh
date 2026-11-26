#ifndef GSPLAT_RASTERIZE_TO_PIXELS_FWD_SUBTILE_CUH
#define GSPLAT_RASTERIZE_TO_PIXELS_FWD_SUBTILE_CUH

// The subtile forward kernel and its launch. Private to
// rasterize_to_pixels_fwd.cu, which includes it (same translation unit).

#include "bindings.h"
#include "raster_alpha.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Forward with subtile masks (PennyGS, arXiv 2611.27525 Sec. 3.4; VkSplat
 * subtile masks, HiGS per-warp hit lists)
 *
 * A 16x16 tile is eight 8x4 subtiles, one per warp. When a batch of Gaussians
 * is staged, each Gaussian also records which subtiles its alpha >=
 * ALPHA_THRESHOLD ellipse can reach (conservative bounding-box test). Each warp
 * compacts the entries reaching its subtile into a hit list, in depth order,
 * and blends only those, with the default kernel's per-pixel step
 * (raster_alpha.cuh). Skipped entries are below the threshold at every pixel
 * of the subtile, which the default kernel skips per pixel, so the outputs are
 * identical; this kernel only decides which entries each warp visits.
 ****************************************************************************/

namespace subtile_fwd {
constexpr uint32_t TILE = 16;
constexpr uint32_t THREADS = TILE * TILE;
// The tile is SUBTILES subtiles of SUB_W x SUB_H pixels, row-major; warp w
// renders subtile w, and bit w of a Gaussian's mask means it may reach it.
constexpr uint32_t SUB_W = 8;
constexpr uint32_t SUB_H = 4;
constexpr uint32_t SUB_COLS = TILE / SUB_W;
constexpr uint32_t SUBTILES = THREADS / 32;
static_assert(SUB_W * SUB_H == 32, "one subtile per warp");
static_assert(SUB_COLS * (TILE / SUB_H) == SUBTILES, "subtiles cover the tile");
static_assert(SUBTILES <= 8, "masks are stored as uint8_t");

// top-left pixel of subtile w, relative to the tile
inline __device__ uint32_t subtile_x0(const uint32_t w) {
    return (w % SUB_COLS) * SUB_W;
}
inline __device__ uint32_t subtile_y0(const uint32_t w) {
    return (w / SUB_COLS) * SUB_H;
}

// Subtiles where gaussian_alpha (raster_alpha.cuh) may accept the Gaussian at
// some pixel centre: bounding box of
// {d : 0.5 d^T Q d <= ln(ALPHA_THRESHOLD_INV * opacity) + err}. It never
// clears a bit the per-pixel test would accept; it may set extra ones. The
// renderer evaluates sigma in float32, where cancellation between the
// quadratic terms can accept pixels the exact ellipse excludes (far,
// anisotropic Gaussians), so `err` bounds that rounding error over the tile:
// 16 * 2^-24 * (largest |term| sum at the tile's corners). Ill-conditioned
// conics (det below 1e-4 * q00 * q11, where det loses its accuracy),
// degenerate or nonfinite input reach every subtile.
//
// A NaN sigma passes gaussian_alpha at MAX_ALPHA whatever the opacity (the
// clamp returns MAX_ALPHA and neither test fires), so the geometry checks come
// before the opacity test: non-finite means or conics, or quadratic terms over
// the tile large enough for sigma to overflow (inf - inf at some pixel), reach
// every subtile.
template <typename S>
inline __device__ uint32_t subtile_mask(
    const S x, const S y, const vec3<S> conic, const S opac, const S tile_x0,
    const S tile_y0
) {
    if (!isfinite(x) || !isfinite(y) || !isfinite(conic.x) ||
        !isfinite(conic.y) || !isfinite(conic.z)) {
        return 0xffu;
    }
    const S dx = fmaxf(fabsf(x - tile_x0), fabsf(x - tile_x0 - (S)TILE));
    const S dy = fmaxf(fabsf(y - tile_y0), fabsf(y - tile_y0 - (S)TILE));
    // bounds each term of sigma at every pixel of the tile; below 1e30 sigma
    // stays finite there
    const S terms = 0.5f * (fabsf(conic.x) * dx * dx + fabsf(conic.z) * dy * dy) +
                    fabsf(conic.y) * dx * dy;
    if (!(terms <= 1e30f)) {
        return 0xffu;
    }
    if (!(opac * ALPHA_THRESHOLD_INV >= 1.f)) {
        return isfinite(opac) ? 0u : 0xffu;
    }
    const S det = conic.x * conic.z - conic.y * conic.y;
    if (!(det > 1e-4f * conic.x * conic.z) || !isfinite(det)) {
        return 0xffu;
    }
    const S tau = __logf(opac * ALPHA_THRESHOLD_INV) * 1.01f + 1e-3f +
                  terms * 9.5367431640625e-7f;
    const S half_x = sqrtf(2.f * tau * conic.z / det) + 0.05f;
    const S half_y = sqrtf(2.f * tau * conic.x / det) + 0.05f;
    if (!isfinite(half_x) || !isfinite(half_y)) {
        return 0xffu;
    }
    uint32_t mask = 0u;
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t w = 0; w < SUBTILES; ++w) {
        // pixel centres of the subtile: x0 + 0.5 ... x0 + SUB_W - 0.5
        const S x0 = tile_x0 + (S)subtile_x0(w);
        const S y0 = tile_y0 + (S)subtile_y0(w);
        const bool reach = x + half_x >= x0 + 0.5f &&
                           x - half_x <= x0 + (SUB_W - 0.5f) &&
                           y + half_y >= y0 + 0.5f &&
                           y - half_y <= y0 + (SUB_H - 0.5f);
        mask |= reach ? (1u << w) : 0u;
    }
    return mask;
}

// Colour staging (PennyGS Sec. 3.4): for up to MAX_STAGE_DIM channels, the
// thread that loads a batch entry also copies its colours to shared memory,
// channel-major ([k][entry]) so the stores are conflict-free and a warp's reads
// of one entry broadcast. The blend arithmetic is unchanged, so outputs stay
// identical. Wider colours are read from global memory per hit: on water2
// (RTX 4090) staging 8, 16 or 32 channels was slower on some views at 1280x720
// or 3840x2160 (the staged batch costs occupancy), while 1-4 channels were
// faster on every view (benchmarks/pennygs/subtile_staging.md).
constexpr uint32_t MAX_STAGE_DIM = 4;
} // namespace subtile_fwd

template <uint32_t COLOR_DIM, typename S>
__global__ void __launch_bounds__(subtile_fwd::THREADS)
    rasterize_to_pixels_fwd_subtile_kernel(
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
        S *__restrict__ render_colors,
        S *__restrict__ render_alphas,
        int32_t *__restrict__ last_ids
    ) {
    using namespace subtile_fwd;
    constexpr bool STAGED = COLOR_DIM <= MAX_STAGE_DIM;
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

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
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

    // a masked tile renders the background with alpha 0 and last_ids 0, as the
    // default kernel; the whole block leaves together, so no warp reaches the
    // ballots below with some lanes exited
    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                render_colors[pix_id * COLOR_DIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
            render_alphas[pix_id] = 0.0f;
            last_ids[pix_id] = 0;
        }
        return;
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches = (range_end - range_start + THREADS - 1) / THREADS;

    // with staged colours the Gaussian ids are not needed
    __shared__ int32_t id_batch[STAGED ? 1 : THREADS];
    __shared__ S color_batch[STAGED ? COLOR_DIM * THREADS : 1]; // [k][entry]
    __shared__ vec3<S> xy_opacity_batch[THREADS];
    __shared__ vec3<S> conic_batch[THREADS];
    __shared__ uint8_t mask_batch[THREADS];
    __shared__ uint16_t hits[SUBTILES][THREADS]; // per-warp hit list of entries

    const S tile_x0 = (S)(tile_col * TILE);
    const S tile_y0 = (S)(tile_row * TILE);
    S T = 1.0f;
    uint32_t cur_idx = 0;
    S pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= THREADS) {
            break;
        }
        const uint32_t batch_start = range_start + THREADS * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            const vec3<S> conic = conics[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conic;
            mask_batch[tr] =
                (uint8_t)subtile_mask<S>(xy.x, xy.y, conic, opac, tile_x0, tile_y0);
            if constexpr (STAGED) {
                const S *c_ptr = colors + g * COLOR_DIM;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    color_batch[k * THREADS + tr] = c_ptr[k];
                }
            } else {
                id_batch[tr] = g;
            }
        }
        block.sync();

        // compact this warp's entries, keeping depth order
        const uint32_t batch_size = min(THREADS, range_end - batch_start);
        uint32_t n_hits = 0;
        for (uint32_t base = 0; base < batch_size; base += 32) {
            const uint32_t e = base + lane;
            const bool hit = e < batch_size && ((mask_batch[e] >> warp_id) & 1u);
            const uint32_t ballot = warp.ballot(hit);
            if (hit) {
                hits[warp_id][n_hits + __popc(ballot & ((1u << lane) - 1u))] = e;
            }
            n_hits += __popc(ballot);
        }
        warp.sync();

        for (uint32_t h = 0; (h < n_hits) && !done; ++h) {
            const uint32_t t = hits[warp_id][h];
            // the default kernel's per-pixel step (raster_alpha.cuh)
            S alpha;
            if (!gaussian_alpha<S>(
                    conic_batch[t], xy_opacity_batch[t], px, py, alpha
                )) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }

            const S vis = alpha * T;
            if constexpr (STAGED) {
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    pix_out[k] += color_batch[k * THREADS + t] * vis;
                }
            } else {
                const S *c_ptr = colors + id_batch[t] * COLOR_DIM;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

// Subtile forward into the preallocated outputs (tile size 16).
template <uint32_t CDIM>
void launch_rasterize_to_pixels_fwd_subtile(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, CDIM] or [nnz, CDIM]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, CDIM]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    torch::Tensor &renders,            // [C, image_height, image_width, CDIM]
    torch::Tensor &alphas,             // [C, image_height, image_width, 1]
    torch::Tensor &last_ids,           // [C, image_height, image_width]
    const cudaStream_t stream
) {
    TORCH_CHECK(
        tile_size == subtile_fwd::TILE,
        "the subtile forward needs tile_size 16, got ",
        tile_size
    );
    const uint32_t C = tile_offsets.size(0);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const dim3 blocks = {C, tile_height, tile_width};
    rasterize_to_pixels_fwd_subtile_kernel<CDIM, float>
        <<<blocks, subtile_fwd::THREADS, 0, stream>>>(
            C,
            flatten_ids.size(0),
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
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
            last_ids.data_ptr<int32_t>()
        );
}

} // namespace gsplat

#endif // GSPLAT_RASTERIZE_TO_PIXELS_FWD_SUBTILE_CUH
