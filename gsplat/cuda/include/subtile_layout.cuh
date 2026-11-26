#ifndef GSPLAT_SUBTILE_LAYOUT_CUH
#define GSPLAT_SUBTILE_LAYOUT_CUH

#include <cooperative_groups.h>
#include <cstdint>

namespace gsplat {

// Layout shared by the subtile forward kernels (3DGS and 2DGS): a 16x16 tile
// is SUBTILES subtiles of SUB_W x SUB_H pixels, row-major; warp w renders
// subtile w, and bit w of a Gaussian's mask means it may reach that subtile.
namespace subtile_layout {
constexpr uint32_t TILE = 16;
constexpr uint32_t THREADS = TILE * TILE;
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

// Writes the batch entries whose mask has bit `warp_id` into `hits`, keeping
// batch (depth) order, and returns their number. All lanes of the warp call it.
inline __device__ uint32_t compact_warp_hits(
    const cooperative_groups::thread_block_tile<32> &warp,
    const uint32_t warp_id,
    const uint8_t *__restrict__ mask_batch,
    const uint32_t batch_size,
    uint16_t *__restrict__ hits
) {
    const uint32_t lane = warp.thread_rank();
    uint32_t n_hits = 0;
    for (uint32_t base = 0; base < batch_size; base += 32) {
        const uint32_t e = base + lane;
        const bool hit = e < batch_size && ((mask_batch[e] >> warp_id) & 1u);
        const uint32_t ballot = warp.ballot(hit);
        if (hit) {
            hits[n_hits + __popc(ballot & ((1u << lane) - 1u))] = e;
        }
        n_hits += __popc(ballot);
    }
    return n_hits;
}
} // namespace subtile_layout

} // namespace gsplat

#endif // GSPLAT_SUBTILE_LAYOUT_CUH
