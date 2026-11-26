#ifndef GSPLAT_RASTER_ALPHA_CUH
#define GSPLAT_RASTER_ALPHA_CUH

#include "types.cuh"

namespace gsplat {

// Per-pixel blending rules of the 3DGS pixel rasterizer, shared by the forward
// kernels in rasterize_to_pixels_fwd.cu (default and subtile). The backward in
// rasterize_to_pixels_bwd.cu repeats them and must stay in step: it recomputes
// alpha from the forward's last_ids and final T.

// alpha is clamped to MAX_ALPHA
constexpr float MAX_ALPHA = 0.999f;
// a Gaussian is skipped at a pixel where alpha < ALPHA_THRESHOLD = 1 / 255;
// subtile_fwd::subtile_mask tests opacity * ALPHA_THRESHOLD_INV >= 1
constexpr float ALPHA_THRESHOLD_INV = 255.f;
constexpr float ALPHA_THRESHOLD = 1.f / ALPHA_THRESHOLD_INV;
// a pixel is done before the Gaussian that would bring T to or below this;
// a double, so `next_T <= TRANSMITTANCE_THRESHOLD` compares in double as before
constexpr double TRANSMITTANCE_THRESHOLD = 1e-4;

// Alpha of a Gaussian (conic; mean and opacity in xy_opac) at the pixel centre
// (px, py), clamped to MAX_ALPHA. Returns false where the Gaussian is skipped:
// negative sigma, or alpha below ALPHA_THRESHOLD.
template <typename S>
inline __device__ bool gaussian_alpha(
    const vec3<S> conic, const vec3<S> xy_opac, const S px, const S py, S &alpha
) {
    const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
    const S sigma =
        0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
        conic.y * delta.x * delta.y;
    alpha = min(MAX_ALPHA, xy_opac.z * __expf(-sigma));
    return !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
}

} // namespace gsplat

#endif // GSPLAT_RASTER_ALPHA_CUH
