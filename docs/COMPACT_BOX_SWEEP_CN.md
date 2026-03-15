# Compact Box `sweep` 路径中文说明

本文用中文解释 `sweep`（快速路径）与 `rect_min`（参考路径）的数学差异，重点讲清 `sweep` 的每个步骤。

---

## 1. 问题定义

对每个投影后的高斯，我们有：

- 中心：`mu = (mu_x, mu_y)`
- 精度矩阵（协方差逆）：

  ```text
  Q = [ q00  q01 ]
      [ q01  q11 ]
  ```

- 局部坐标：

  ```text
  dx = x - mu_x
  dy = y - mu_y
  ```

- 二次型：

  ```text
  f(x, y) = [dx dy] * Q * [dx dy]^T
  ```

Compact Box 的保留条件是：

```text
f(x, y) <= tau2
```

其中 `tau2` 可由不透明度推导：

```text
tau2_i = compact_box_mult * 2 * log(opacity_i * 255)
```

这定义了一个椭圆（`Q` 正定时）。目标是快速求出它覆盖了哪些 tile。

---

## 2. `sweep` 的核心思想

`rect_min` 是逐 tile 精确判定：每个 tile 都做一次最小值优化。

`sweep` 则是几何扫描：

1. 先算椭圆包围范围；
2. 选 x 或 y 方向做条带扫描；
3. 每个条带解“直线与椭圆”的交点；
4. 得到该条带覆盖的连续 tile 区间；
5. 直接计数/写出整段区间。

换句话说：`sweep` 是“scanline/span fill”思想，不是“逐 tile 优化”。

---

## 3. 步骤一：由 `Q` 与 `tau2` 得椭圆外接范围

先算：

```text
det = q00 * q11 - q01 * q01
x_ext = sqrt(tau2 * q11 / det)
y_ext = sqrt(tau2 * q00 / det)
```

于是连续空间包围盒：

```text
x in [mu_x - x_ext, mu_x + x_ext]
y in [mu_y - y_ext, mu_y + y_ext]
```

再转成 tile 索引范围（`floor/ceil` + clamp 到图像边界）。

实现里还会与 coarse box（由半径给出的候选范围）取交，保证更稳健。

---

## 4. 步骤二：选择扫描方向（扫短边）

计算 tile 跨度：

```text
x_span = rect_max_x - rect_min_x
y_span = rect_max_y - rect_min_y
```

若 `y_span < x_span`，按 y 扫；否则按 x 扫。

原因很直接：扫短边能减少循环次数，通常更快。

---

## 5. 步骤三：每个条带求椭圆交点（解析解）

以下以“按 x 扫”为例（按 y 完全对称）。

对每个 x-tile 条带：

```text
x0 = tx * tile_size
x1 = x0 + tile_size
```

固定 `x`，解方程 `f(x, y) = tau2`，得到 y 方向交点：

```text
y = mu_y - (q01 * dx) / q11 +- sqrt(q11 * tau2 - det * dx^2) / q11
```

其中 `dx = x - mu_x`。

实现会对 `x0`、`x1` 都算一次，形成条带在 y 方向的候选覆盖范围。

另外，条带内部可能包含椭圆的局部极值点（不一定落在边界），所以还要补上极值检查，避免漏掉最大/最小 y。

---

## 6. 步骤四：连续范围离散化为 tile 区间

若得到连续覆盖 `[v_min, v_max]`（在 y 轴上），转换为离散 tile 行：

```text
ty_min = floor(v_min / tile_size)
ty_max = floor(v_max / tile_size) + 1
```

最后得到半开区间：

```text
ty in [ty_min, ty_max)
```

这是一段连续 tile，可一次处理，而不是逐格判定。

---

## 7. 步骤五：两遍一致的 count / emit

`isect_tiles` 是两遍流程：

- Pass 1：只计数（prefix-sum 用）
- Pass 2：按同样逻辑写出 `(camera, tile, depth, flatten_id)`

`sweep` 在两遍复用同一套条带区间逻辑，这样可保证：

- 计数与实际写出严格一致
- 不会出现偏移错位

---

## 8. 为什么 `sweep` 更快

`rect_min` 的复杂度更接近“候选 tile 数量 * 每 tile 计算成本”。

`sweep` 更像“条带数量 * 解析求交 + 连续区间写出”。

当椭圆覆盖 tile 较多时，`sweep` 通常显著减少计算和分支开销。

---

## 9. 为什么边界可能有轻微差异

与 `rect_min` 相比，`sweep` 可能在边缘 tile 上有极少差异，主要来自：

1. 条带边界采样（`x0/x1` 或 `y0/y1`）
2. `floor/ceil/+1` 离散化规则
3. 边界 clamp（图像边界或 coarse box）
4. 浮点误差

这些差异通常只出现在椭圆边缘，画面影响很小，但性能收益明显。

---

## 10. 实践建议

- 追求速度：优先用 `compact_box_impl="sweep"`
- 做基线对照/调试：用 `compact_box_impl="rect_min"`
- 若要比较质量差异，配合 benchmark 输出的 `max_abs/psnr` 观察边界影响
