#ifndef GSPLAT_WARP_REDUCE_SCATTER_CUH
#define GSPLAT_WARP_REDUCE_SCATTER_CUH

#include <cooperative_groups.h>

namespace gsplat {

namespace detail {

// Halve the live component window while reducing across successively smaller
// lane groups. Each surviving lane owns one component, rather than replicating
// every reduced component in all 32 lanes. All lanes must participate.
template <int DIM, int STRIDE> struct WarpReduceScatterStep {
    static_assert(DIM > 1 && DIM <= STRIDE, "Invalid scatter window/stride");
    template <typename S>
    static inline __device__ S run(
        const S (&values)[DIM], int lane, int &component, bool &owner
    ) {
        constexpr int HALF = (DIM + 1) / 2;
        const bool upper = (lane & STRIDE) != 0;
        S next[HALF];
#pragma unroll
        for (int i = 0; i < HALF; ++i) {
            const S lo = values[i];
            const S hi = i + HALF < DIM ? values[i + HALF] : S(0);
            const S send = upper ? lo : hi;
            const S kept = upper ? hi : lo;
            next[i] = kept + __shfl_xor_sync(0xffffffffu, send, STRIDE);
        }
        if (upper) {
            component += HALF;
        }
        // A non-power-of-two window has padded upper components. Preserve
        // their invalidity through subsequent stages instead of aliasing IDs.
        const int previous_component = component;
        S result = WarpReduceScatterStep<HALF, STRIDE / 2>::run(
            next, lane, component, owner
        );
        if (upper && component - previous_component >= DIM - HALF) {
            owner = false;
        }
        return result;
    }
};

template <int STRIDE> struct WarpReduceScatterStep<1, STRIDE> {
    static_assert(STRIDE >= 1 && (STRIDE & (STRIDE - 1)) == 0,
                  "Invalid terminal scatter stride");
    template <typename S>
    static inline __device__ S run(
        const S (&values)[1], int lane, int &component, bool &owner
    ) {
        S result = values[0];
#pragma unroll
        for (int offset = STRIDE; offset > 0; offset /= 2) {
            result += __shfl_xor_sync(0xffffffffu, result, offset);
        }
        owner = owner && ((lane & (2 * STRIDE - 1)) == 0);
        return result;
    }
};

template <typename S> struct WarpReducedComponent {
    S value;
    int component;
    bool owner;
};

} // namespace detail

// Entry contract: all 32 lanes of a complete warp must call together for the
// same Gaussian. Partial-warp blocks must use the caller's standard fallback.
// Only an owner's value/component may be used for writeback. Initialize the
// recursive ownership state here, not at each call site.
template <int DIM, typename S>
inline __device__ detail::WarpReducedComponent<S> warp_reduce_scatter(
    const S (&values)[DIM], const cooperative_groups::thread_block_tile<32> &warp
) {
    static_assert(DIM >= 1 && DIM <= 16,
                  "Full-warp scatter supports 1 to 16 gradient components");
    int component = 0;
    bool owner = true;
    const S sum = detail::WarpReduceScatterStep<DIM, 16>::run(
        values, warp.thread_rank(), component, owner
    );
    return {sum, component, owner};
}

} // namespace gsplat

#endif
