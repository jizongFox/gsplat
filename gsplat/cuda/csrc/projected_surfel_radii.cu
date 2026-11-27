#include "bindings.h"

#include <cuda_runtime.h>

namespace gsplat {

__global__ void projected_surfel_radii_kernel(
    const float* transforms, const int32_t* raw_radii, int32_t* output,
    int64_t n, int width, int height, float radius_clip) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* h = transforms + i * 9;
    if (raw_radii[i] <= 0 || !isfinite(h[8]) || h[8] <= 0.0f) {
        output[i] = 0;
        return;
    }
    const float iz = 1.0f / h[8];
    const float x = h[2] * iz, y = h[5] * iz;
    const float cx = fminf(fmaxf(x, -0.15f * width), 1.15f * width);
    const float cy = fminf(fmaxf(y, -0.15f * height), 1.15f * height);
    const float u = (h[0] - cx * h[6]) * iz;
    const float v = (h[1] - cx * h[7]) * iz;
    const float p = (h[3] - cy * h[6]) * iz;
    const float q = (h[4] - cy * h[7]) * iz;
    const float a = u * u + v * v, b = u * p + v * q, d = p * p + q * q;
    const float delta = a - d;
    const float lambda_max = 0.5f * (a + d + sqrtf(delta * delta + 4.0f * b * b));
    const float radius = 3.0f * sqrtf(fmaxf(lambda_max, 0.0f));
    const bool keep = isfinite(radius) && radius > radius_clip &&
        x + radius > 0.0f && x - radius < width &&
        y + radius > 0.0f && y - radius < height;
    output[i] = keep ? static_cast<int32_t>(ceilf(radius)) : 0;
}

torch::Tensor projected_surfel_radii_tensor(
    const torch::Tensor& ray_transforms, const torch::Tensor& raw_radii,
    uint32_t width, uint32_t height, float radius_clip) {
    GSPLAT_DEVICE_GUARD(ray_transforms);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(raw_radii);
    TORCH_CHECK(ray_transforms.scalar_type() == torch::kFloat32);
    TORCH_CHECK(raw_radii.scalar_type() == torch::kInt32);
    TORCH_CHECK(ray_transforms.dim() == 3 && ray_transforms.size(1) == 3 &&
                ray_transforms.size(2) == 3);
    TORCH_CHECK(raw_radii.dim() == 1 && raw_radii.size(0) == ray_transforms.size(0));
    auto output = torch::zeros_like(raw_radii);
    const int64_t n = raw_radii.size(0);
    if (n) {
        projected_surfel_radii_kernel<<<(n + 255) / 256, 256>>>(
            ray_transforms.data_ptr<float>(), raw_radii.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), n, width, height, radius_clip);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "projected_surfel_radii kernel launch failed");
    }
    return output;
}

}  // namespace gsplat
