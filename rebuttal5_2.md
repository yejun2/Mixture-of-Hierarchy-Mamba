**Response to Reviewer mhMg**

We thank the reviewer for the positive assessment and for expressing support for publication. We particularly appreciate the questions regarding practical execution efficiency, the valid depth of the hierarchy, and complete inference cost. We agree that reduced arithmetic computation should be clearly distinguished from realized wall-clock performance.

**Q1–Q2. Routing overhead, GPU fragmentation, and wall-clock efficiency**

**Response**

We agree that the current dynamically routed implementation introduces non-negligible system overhead. The routing controller itself is lightweight: it adds only 0.666M parameters, corresponding to 0.52% of MoH-Base, and requires 0.0015 GFLOPs per sample, below 0.002% of the total computation. Therefore, the main overhead does not arise from computing the routing decisions themselves, but from executing the dynamically selected computation paths.

In the current PyTorch implementation, route-dependent stage and depth selection requires feature gathering, branch-specific execution, and restoration of the original batch layout. Profiling shows that MoH produces 3,726 CUDA events and 3,191 `cudaLaunchKernel` calls per forward pass. The average CUDA event duration is approximately 31 μs, indicating that the model executes a large number of relatively small GPU operations. The main routing-boundary operations include `nonzero`, `index_select`, `index_copy_`, `scatter_`, and `topk`. The current execution also invokes `aten::item` and `aten::_local_scalar_dense` 159 times each and triggers 258 `cudaMemcpyAsync` operations.

This fragmentation explains why the reduction in FLOPs does not currently translate into faster wall-clock execution. MoH-Base requires 80.88 GFLOPs per denoiser evaluation, compared with 97.52 GFLOPs for ZigMa, but its measured latency is higher. Under the same RTX A6000 and AMP setting, MoH-Base requires 49.71 ms at batch size 1 and 87.24 ms at batch size 8. For reference, ZigMa requires 19.54 ms and 61.93 ms under the corresponding settings.

We therefore will revise our efficiency claim to reduced active arithmetic computation and parameter footprint, rather than faster wall-clock inference in the current implementation. Grouped route execution and fused dispatch may reduce the identified fragmentation, but we will report the current latency limitation explicitly rather than relying on future optimization.

**Q3–Q4. Valid hierarchy depth, excessive compression, and architectural stability**

**Response**

Table 5 directly evaluates hierarchy depth under the same training setting. The two-stage configuration (32, 16) achieves an FID of 13.17, while extending the hierarchy to (32, 16, 8) improves the FID to 11.93. This shows that the additional lower-resolution stage provides useful global context.

However, further extending the hierarchy to (32, 16, 8, 4) degrades the FID to 12.64. Thus, increasing hierarchy depth is not universally beneficial. For the 32 × 32 latent representation used in our experiments, 32 → 16 → 8 provides the best observed trade-off between global context aggregation and preservation of spatial information. The additional 4 × 4 stage leaves only 16 spatial positions and introduces excessive compression.

Although the decoder reintegrates higher-resolution features through gated fusion and skip connections, these paths do not fully compensate for the degradation caused by the additional compression. We emphasize that Table 5 demonstrates quality degradation at the deeper stage; it does not indicate numerical divergence or general training instability.

Accordingly, all main experiments use the three-stage (32, 16, 8) hierarchy, and the 4 × 4 stage is excluded from the routing candidate set.

**Q5–Q6. Time per denoising step and complete inference cost**

**Response**

For MoH-Base, one denoiser forward pass requires 49.71 ms on a single NVIDIA RTX A6000 with AMP and batch size 1. At batch size 8, the corresponding latency is 87.24 ms. We will report these per-evaluation latency values explicitly instead of only reporting steps per second.

The final evaluation uses Dopri5, an adaptive Runge–Kutta ODE solver. Therefore, `num_steps=250` denotes the solver configuration and does not imply exactly 250 denoiser evaluations. In the measured adaptive-*k* MoH run, the solver used 74 network function evaluations (NFEs).

**Limitation. Practical execution and extreme hierarchy depth**

**Response**

We agree that hierarchy depth cannot be increased without restriction. Table 5 shows that extending (32, 16, 8) to (32, 16, 8, 4) degrades FID from 11.93 to 12.64, suggesting that the information loss caused by compression to 4 × 4 outweighs the benefit of additional global aggregation.

MoH mitigates reliance on a single fixed bottleneck through adaptive-*k* stage selection. At each timestep, SSR selects a subset of the predefined candidate stages, and the lowest-resolution selected stage determines the decoder bottleneck. This allows the model to adapt its abstraction level throughout denoising rather than always using the deepest stage.

For the 32 × 32 latent resolution considered in this work, (32, 16, 8) provides the best observed trade-off. The candidate hierarchy itself remains predefined, and extending it appropriately to substantially higher resolutions remains future work. We will state this limitation in the main paper.
