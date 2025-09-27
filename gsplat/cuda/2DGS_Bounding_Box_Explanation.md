# Bounding Box Computation in 2D Gaussian Splatting Code

The bounding box computation in the `_fully_fused_projection_2dgs` function derives an axis-aligned bounding box (AABB)
for each projected 2D Gaussian splat in screen space. This process calculates the projected center (`means2d`) and
half-extents (`extents`) of the ellipse resulting from the perspective projection of the local unit circle, representing
the Gaussian disk in tangent space. The approach leverages projective geometry to compute these values efficiently
without explicit matrix inversion, ensuring numerical stability and computational efficiency. The final radius is
computed as the ceiling of three times the maximum half-extent, providing a conservative bound suitable for tile-based
rasterization and culling, accounting for the Gaussian falloff (approximately 3σ coverage for `exp(-(u² + v²)/2)`).

## Mathematical Foundation

The local 2D Gaussian is modeled as a unit circle in the tangent plane, defined by the conic equation \( h_l^T Q h_l =
0 \), where \( h_l = [u, v, 1]^T \) is the homogeneous coordinate in local space, and \( Q = \text{diag}(1, 1, -1) \)
represents the quadric \( x^2 + y^2 - 1 = 0 \). The projection to screen space uses the transformation matrix \( T_
{sl} \), which maps from local tangent space to screen space, yielding the projected conic:

\[
C_{\text{screen}} = T_{sl}^{-T} Q T_{sl}^{-1}.
\]

To avoid inverting \( T_{sl} \), the code works with the inverse conic \( S = C_{\text{screen}}^{-1} = T_{sl} Q^{-1} T_
{sl}^T \), where \( Q^{-1} = Q \) due to its diagonal form. The center and extents of the projected ellipse are
extracted using polar duality:

- **Projected Center**: The center \( \text{means2d} = [u_0, v_0] \), where:
  \[
  u_0 = \frac{S_{02}}{S_{22}}, \quad v_0 = \frac{S_{12}}{S_{22}}.
  \]
- **Half-Extents**: The extents \( \text{extents} = [\Delta u, \Delta v] \), where:
  \[
  \Delta u = \sqrt{u_0^2 - \frac{S_{00}}{S_{22}}}, \quad \Delta v = \sqrt{v_0^2 - \frac{S_{11}}{S_{22}}}.
  \]

These formulas derive from the tangency conditions for axis-parallel lines to the ellipse. For vertical lines (
constant \( u \)), the dual condition \( l^T S l = 0 \) with \( l = [1, 0, -u]^T \) yields a quadratic equation whose
roots define the minimum and maximum \( u \), with the half-width \( \Delta u \) as shown. The same applies for \( v \).
This ensures an exact AABB for the projected ellipse.

Given \( M = T_{sl}^T \) in the code, the inverse conic becomes:

\[
S = M Q^{-1} M^T,
\]

where \( M \) aligns with the paper's notation \( M = (WH)^T \), with \( W \) as the world-to-screen transformation
and \( H \) encoding the Gaussian's geometry. The conic elements are computed as:

- \( S_{22} = e_3^T S e_3 = (M^T e_3)^T Q^{-1} (M^T e_3) \), using the third row of \( M \).
- Similar dot products yield \( S_{00}, S_{11}, S_{02}, S_{12} \).

## Code Breakdown with Mathematical Integration

The code approximates the full perspective projection using an affine transformation \( T_{sl} = K @ T_{cl} \), where \(
T_{cl} \) maps from local to camera space and \( K \) is the intrinsic matrix. This approximation is sufficient for
bounding while avoiding division instability. Below is a step-by-step breakdown of the code with corresponding
mathematical operations:

1. **Transformation Matrix Computation**:
    - `RS_wl = _quat_scale_to_matrix(quats, scales)` computes the 3×3 rotation-scaling matrix in world space for each
      Gaussian.
    - `RS_cl = torch.einsum("cij,njk->cnik", R_cw, RS_wl)` transforms it to camera space using the camera rotation \( R_
      {cw} \).
    - `T_cl = torch.cat([RS_cl[..., :2], means_c[..., None]], dim=-1)` assembles the 3×3 matrix mapping
      local \( [u, v, 1]^T \) to camera coordinates, incorporating the Gaussian's center \( \text{means_c} \).
    - `T_sl = torch.einsum("cij,cnjk->cnik", Ks[:, :3, :3], T_cl)` applies the intrinsic matrix \( K \) to map to screen
      homogeneous coordinates.
    - `M = torch.transpose(T_sl, -1, -2)` computes \( M = T_{sl}^T \), aligning with the paper's notation for plane
      transformations.

2. **Normal Computation**:
    - `normals = RS_cl[..., 2]` extracts the normal vector in camera space (third column of \( RS_{cl} \)),
      corresponding to the disk's orientation.
    - The sign is adjusted (`multiplier`) to ensure front-facing normals (\( \cos > 0 \)) by checking the dot product
      with the camera-space mean.

3. **Conic Element Computation**:
    - `test = torch.tensor([1.0, 1.0, -1.0], device=means.device).reshape(1, 1, 3)` represents \( \text{diag}(Q) \),
      encoding the local unit circle.
    - `d = (M[..., 2] * M[..., 2] * test).sum(dim=-1, keepdim=True)` computes \( S_{22} \) (or equivalently \( C_
      {\text{screen}}[2,2] \) in dual form), the normalization factor for the conic. If \( |d| \leq \epsilon \), the
      Gaussian is degenerate (e.g., viewed edge-on) and marked invalid.
    - `f = torch.where(valid, test / d, torch.zeros_like(test)).unsqueeze(-1)` normalizes the dual contributions for
      center and extent calculations.

4. **Projected Center (means2d)**:
    - `means2d = (M[..., :2] * M[..., 2:3] * f).sum(dim=-2)` computes \( [u_0, v_0] \) as the weighted dot products,
      equivalent to \( (S_{02} / S_{22}, S_{12} / S_{22}) \). The multiplication incorporates the third row's
      contribution to the off-diagonal terms of \( S \).

5. **Half-Extents (extents)**:
    - `temp = (M[..., :2] * M[..., :2] * f).sum(dim=-2)` computes \( S_{00} / S_{22} \) and \( S_{11} / S_{22} \).
    - `extents = torch.sqrt(means2d**2 - temp)` calculates the half-widths \( \Delta u \) and \( \Delta v \),
      representing the spread from the center along each axis.

6. **Radius and Validity**:
    - `radius = torch.ceil(3.0 * torch.max(extents, dim=-1).values)` computes the bounding radius in pixels, taking the
      maximum of \( \Delta u \) and \( \Delta v \) and scaling by 3 to cover the Gaussian's falloff.
    - Validity checks ensure the Gaussian's depth is within the near and far planes (`depths > near_plane` and
      `depths < far_plane`) and that the AABB lies within the image bounds (`means2d ± radius` within `[0, width]` and
      `[0, height]`). Invalid Gaussians have `radius = 0`.

## Summary

The resulting AABB for each Gaussian is defined as:

\[
[\text{means2d}[0] - \text{radius}, \text{means2d}[0] + \text{radius}] \times [\text{means2d}[1] - \text{radius},
\text{means2d}[1] + \text{radius}],
\]

where `radius = ceil(3.0 * max(extents))` ensures a conservative bound due to the maximum and multiplier. This
facilitates efficient rendering by limiting per-pixel evaluations to relevant splats, aligning with the paper's
tile-based rasterization strategy. While the paper uses the full ray-splat intersection method (Equation 10) for exact
rendering, the bounding box computation relies on this conic-based approach for speed, using an affine approximation of
the perspective projection.