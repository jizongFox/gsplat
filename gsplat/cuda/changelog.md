# Summary: `v_means2d` vs `v_densify` in 2DGS

## What they are

### `v_means2d`

- The gradient of the loss w.r.t. the projected 2D mean used by the *2D projected Gaussian* branch.
- Computed as `v_xy_local` in the backward kernel only when the forward selected the 2D Gaussian weight (case 2).
- **File/lines:**
    - `gsplat/cuda/csrc/rasterize_to_pixels_2dgs_bwd.cu`:
        - 508–519 (compute)
        - 586–589 (accumulate)
        - 590–594 (`v_means2d_abs` if AbsGrad enabled)

### `v_densify`

- A distilled densification signal derived from the gradient w.r.t. the 3×3 pixel→splat transform matrix
  `ray_transforms`.
- Specifically takes the transform entries that directly shift the projected center in screen space, scaled by depth:  
  v_densify[0] = v_ray_transforms[0,2] * depth
  v_densify[1] = v_ray_transforms[1,2] * depth

yaml
Copy code

- **File/lines:**
- `gsplat/cuda/csrc/rasterize_to_pixels_2dgs_bwd.cu`: 600–605

---

## Where they come from

### `v_means2d` path (2D projected Gaussian branch, *case 2*)

- Forward chose the 2D projected Gaussian weight (when `gauss_weight_2d` dominates).
- Backward computes:
  v_G = opac * v_alpha # alpha blending / visibility / transmittance included
  v_xy_local = v_G * (-vis * FILTER_INV_SQUARE * d)

go
Copy code

- **File/lines:**
- `rasterize_to_pixels_2dgs_bwd.cu`: 474–477, 508–519

### `v_densify` path (geometry/transform branch, *case 1*)

- Forward used exact ray–primitive intersection (when `gauss_weight_3d` dominates).
- Backward computes `v_u_M_local`, `v_v_M_local`, `v_w_M_local` from:
  v_G = opac * v_alpha # includes visibility, transmittance T, geometry, etc.

markdown
Copy code

- These accumulate into `v_ray_transforms` (9 entries), then `v_densify` picks `[0,2]` and `[1,2]` scaled by depth.
- **File/lines:**
- `rasterize_to_pixels_2dgs_bwd.cu`: 479–507 (compute), 575–584 (accumulate), 600–605 (`v_densify`)

---

## How alpha blending is included

- Both signals inherit alpha/visibility/transmittance effects:
- `v_alpha` is built with color/normal/alpha gradients under alpha compositing and transmittance updates (
  `T *= 1/(1 - alpha)`).
- `v_G = opac * v_alpha` multiplies opacity and is used in both branches before computing:
    - `v_xy_local` (case 2)
    - `v_u_M` / `v_v_M` / `v_w_M` (case 1)
- **File/lines:**
- `rasterize_to_pixels_2dgs_bwd.cu`:
    - 395–461 (`T`, `fac`, `v_alpha`)
    - 474–477 (`v_G = opac * v_alpha`)

---

## Key differences

| Aspect              | `v_means2d`                                                                                                   | `v_densify`                                                                                                                       |
|---------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Path activation** | Nonzero only when the 2D projected Gaussian branch is active (case 2). Zero when the geometry branch is used. | Derived from the geometry/transform branch (case 1) via `v_ray_transforms`.                                                       |
| **Quantity**        | Direct `dL/d(projected 2D mean)` from the 2D projected Gaussian approximation.                                | A proxy for center-shift sensitivity, extracted from `dL/d(ray_transforms)` by selecting `T[0,2]`, `T[1,2]` and scaling by depth. |
| **Units/usage**     | Pixel-space gradient on the 2D mean variable. Also supports AbsGrad as `v_means2d_abs`.                       | A heuristic densification metric, not a primary autograd output. Exposed as `meta["gradient_2dgs"]` and used by strategies.       |

---
