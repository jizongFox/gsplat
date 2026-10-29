# 3DGS CUDA 优化教程：从文章到 gsplat

这是一份面向学习的中文教程。它解释文章
[What the 3DGS Literature Left on the Table](https://grgkopanas.github.io/3dgs_cuda_blog/)
中的优化思想，并把它们映射到本次 `gsplat` 实验。

> 这不是把文章逐字翻译成中文，也不是声称所有优化都已经移植。
> 文中会明确标记：**已实现**、**已实验但回滚**、**只做了源码分析**。

## 阅读顺序

1. [数学基础与 CUDA 模型（LaTeX 章节）](chapters_math.tex)：参数化、投影 Jacobian、conic、compositing、VJP、warp/SIMT。
2. [文章 S00–S06 与 negative results（LaTeX 章节）](chapters_article.tex)：逐项讲动机、公式、代码思路、失败风险和 fork 里的状态。
3. [可手算/运行的例子（LaTeX 章节）](chapters_examples.tex)：alpha 数值、warp reduction、predication、Amdahl 和误差度量。
4. [gsplat 代码与本机实测（LaTeX 章节）](chapters_evaluation.tex)：source map、tile4 partial-warp bug、验证矩阵、完整 timings。

## PDF 教材版

- [中文 LaTeX PDF](gsplat_cuda_tutorial.pdf)
- [LaTeX 源文件](gsplat_cuda_tutorial.tex)
- [数学与 CUDA 基础章节](chapters_math.tex)
- [S00–S06 逐点讲解章节](chapters_article.tex)
- [手算与代码例子章节](chapters_examples.tex)
- [gsplat 移植与本机评估章节](chapters_evaluation.tex)

PDF 使用 `lualatex`、Noto CJK 中文字体、TikZ 流程图和 AMS 数学公式生成。新版是 **47 页 A4 教材**，由四个长章节组成；先前 4 页版只是摘要，已经被本版取代。内容逐个解释原文 S00–S06、文献优化、负结果、数学推导、数值手算、可运行 Python、CUDA 伪代码、失败边界和 RTX 4090 的实测；同时明确标记已移植、已回滚和未实现内容。

## 最重要的结论

本次最终保留了三个互相配合的后向优化：

1. **warp reduce-scatter**：减少梯度归约 shuffle，并让多个 lane 并行写原子加。
2. **后向遍历尾部裁剪**：利用前向保存的 `last_ids`，不再加载整段完全被遮挡的尾部。
3. **安全的 branchless/predicated backward**：窄通道路径对无效像素使用恒等更新，但不执行可能产生 `0 * Inf` 的梯度算术。

内核实现提交：`88513ea`。本教程和 benchmark 资料也随 gsplat 仓库维护，不需要 Gaussian-splatting 父仓库中的 Python 包装或 gitlink 变更。

在给定的 20,789,814 个 Gaussian、RTX 4090 和 1280×720 相机上：

- raster-only RGB 加速约 **1.34×–1.65×**；
- RGB+ED + absgrad 在 camera 0 上约 **1.31× raster / 1.08× full render**；
- 136 个回归案例、默认/非默认 CUDA stream、memcheck、synccheck 均通过；
- 单次完整路径的 quaternion 梯度相对 L2 仍约为 `1.63e-4`，超过严格 `1e-4` 门槛；这被明确记录为限制，而没有调高阈值掩盖。

完整的工程实验日志仍在：
`benchmarks/blog_eval/REPORT.md`。

PDF 源文件通过 `\input{chapters_math.tex}` 等章节拼成；章节只含 document body，可在 TeX 编辑器中逐章阅读。主文件已提供中文字体、数学、代码、表格与 TikZ 所需 preamble。
