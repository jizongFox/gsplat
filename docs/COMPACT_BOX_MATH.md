# Compact Box Math: `rect_min` vs `sweep`

This note explains the math behind Compact Box (CB) tile pruning, and why
`rect_min` and `sweep` differ in speed while staying close in behavior.

---

## 1) Core CB Quantity

For each projected Gaussian, we work in image space with:

- Center: `mu = (mu_x, mu_y)`
- Precision (inverse covariance) matrix:

  ```text
  Q = [ q00  q01 ]
      [ q01  q11 ]
  ```

- Local coordinates:

  ```text
  dx = x - mu_x
  dy = y - mu_y
  ```

- Quadratic form:

  ```text
  f(x, y) = [dx dy] * Q * [dx dy]^T
  ```

Interpretation:

- `f(x, y)` is the squared Mahalanobis distance under `Q`.
- Small `f` means near the Gaussian center in anisotropic metric.
- Large `f` means far away.

CB keeps tile contributions where this distance is below threshold.

---

## 2) Threshold and Opacity Coupling

If no explicit global override is given, threshold is per Gaussian:

```text
tau2_i = compact_box_mult * 2 * log(opacity_i * 255)
```

The keep region is:

```text
f(x, y) <= tau2_i
```

### Intuition

- Higher opacity -> larger `tau2_i` -> larger ellipse -> more tiles kept.
- Lower opacity -> smaller `tau2_i` -> tighter ellipse -> fewer tiles kept.
- If `tau2_i <= 0`, the Gaussian contributes zero CB tiles.

### Validity checks

CB hard-skips a Gaussian if conic is invalid:

- `q00 <= 0`, or
- `q11 <= 0`, or
- `q00 * q11 - q01 * q01 <= 0` (not positive-definite).

---

## 3) Geometric Picture

The set

```text
{ (x, y) | f(x, y) <= tau2 }
```

is an ellipse in image space (possibly rotated when `q01 != 0`).

Tile pruning asks: does this ellipse intersect a tile?

---

## 4) `rect_min` vs `sweep`: Same Ellipse, Different Intersection Operator

## `rect_min` (reference path)

For each candidate tile rectangle `R`, compute:

```text
min_{(x, y) in R} f(x, y)
```

Keep tile iff:

```text
min_{(x, y) in R} f(x, y) <= tau2
```

In implementation, this minimum is evaluated via finite convex candidates:

- 4 corners
- interior optimum `(0, 0)` if it lies inside the tile-local rectangle
- edge stationary points:
  - `y* = -q01 * x / q11` on vertical edges
  - `x* = -q01 * y / q00` on horizontal edges

Pros:

- Tight per-tile test.

Cons:

- Expensive, because it is done tile-by-tile.

---

## `sweep` (fast path)

Instead of per-tile optimization, it computes ellipse spans per strip:

1. Build ellipse extent bounds from `Q` and `tau2`.
2. Sweep by x-slices (or y-slices, whichever span is shorter).
3. For each slice, solve ellipse-line intersections analytically.
4. Convert continuous min/max span to a contiguous tile index range.
5. Emit/count full range directly.

This is a scanline/span fill of the same ellipse.

Pros:

- Lower overhead and better throughput.

Cons:

- Boundary discretization can differ slightly from exact per-tile min test.

---

## 5) Why Small Output Differences Can Exist

Both methods target the same set `f <= tau2`, but:

- `rect_min` is exact rectangle-min criterion per tile.
- `sweep` uses slice boundary intersections plus range discretization.

So a few borderline tiles may differ near ellipse boundaries.

In practice this often gives negligible image deltas while improving runtime.

---

## 6) Practical Relationship Summary

```text
opacity up -> tau2 up -> ellipse area up -> n_isects up -> more raster work
```

```text
Q larger in precision sense -> ellipse tighter -> n_isects down
```

```text
compact_box_mult up -> tau2 up -> less pruning
```

---

## 7) Rule of Thumb

- Use `sweep` for performance runs.
- Use `rect_min` as a reference/debug path when you want stricter per-tile
  geometric checking.
