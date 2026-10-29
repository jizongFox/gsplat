#ifndef GSPLAT_WARP_REDUCE_SCATTER_CUH
#define GSPLAT_WARP_REDUCE_SCATTER_CUH

namespace gsplat {

// Halve the live component window while reducing across successively smaller
// lane groups. Each surviving lane owns one component, rather than replicating
// every reduced component in all 32 lanes. All lanes must participate.
template <int DIM, int STRIDE> struct WarpReduceScatter {
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
        S result = WarpReduceScatter<HALF, STRIDE / 2>::run(
            next, lane, component, owner
        );
        if (upper && component - previous_component >= DIM - HALF) {
            owner = false;
        }
        return result;
    }
};

template <int STRIDE> struct WarpReduceScatter<1, STRIDE> {
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

} // namespace gsplat

#endif
