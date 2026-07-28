We thank the reviewer for recognizing the substantial performance gains and the alignment between timestep-conditioned routing and the coarse-to-fine nature of diffusion.

**W1. Recent baselines**

**Response**

We agree that recent Mamba-based diffusion models should be better covered. We therefore add U-Shape Mamba, the most directly related recent Self-attention-free Mamba baseline evaluated on MS-COCO at 256 × 256.

*Conditional image generation on MS-COCO at 256 × 256.*

| Method | Backbone | FID<sup>5k</sup> ↓ |
| :--- | :---: | :---: |
| VisionMamba | Mamba | 60.20 |
| ZigMa | Mamba | 41.80 |
| U-Shape Mamba | Mamba | 39.10 |
| MoH (ours) | Mamba | **20.00** |

U-Shape Mamba forms a U-shaped hierarchy by reducing the sequence after Mamba processing and reconstructing it in the decoder through upsampling and skip connections. In contrast, MoH treats each spatial resolution as a distinct Mamba scanning space, so the scanning process itself is performed hierarchically across multiple spatial scales.

We will also include hybrid models such as Dimba and DiMSUM as broader references. These methods combine Mamba with Transformer or self-attention modules to maximize overall backbone performance, whereas MoH focuses on improving the Mamba architecture itself without self-attention. Thus, Self-attention-free Mamba models provide the most direct controlled comparison, while hybrid models provide complementary performance context.

**W2. Motivation for dynamic architectural routing in Mamba**

**Response**

We thank the reviewer for this important question. Our motivation was to design a self-attention-free Mamba backbone that is better suited to vision.

Inductive bias is central to visual architecture design. CNNs explicitly encode locality and spatial hierarchy, while Transformers can model broad spatial relationships through direct token interactions. Mamba, however, was originally designed for one-dimensional sequences and does not inherently provide explicit multi-scale spatial representations for visual tokens. Most prior vision Mamba works mainly address this issue by modifying the scan strategy.

To address this limitation, we introduce spatially hierarchical Mamba scanning spaces, where Mamba operates at multiple resolutions. This provides the backbone with explicit multi-scale spatial structure. However, a fixed hierarchy cannot determine how strongly features should be mixed and compressed at each denoising timestep. We therefore formulate this process as the CMR routing space, allowing spatial mixing and compression to adapt to the coarse-to-fine denoising pattern [7].

We further exploit the hierarchical structure of MoH when designing EDR. Higher-resolution stages are better suited to preserving local details, whereas lower-resolution stages favor more aggregated semantic representations, reflecting the coarse-to-fine structure of diffusion. Based on this observation, EDR adapts the Mamba depth within each stage so that the degree of information aggregation can be adjusted according to both the spatial level and the denoising timestep.

Thus, the hierarchical structure and CMR are motivated by the lack of explicit spatial inductive bias in Mamba, while EDR is motivated by the behavior of Mamba state propagation and coarse-to-fine pattern. We agree that the broader principle of timestep-dependent architectural routing may also be extended to other backbones. For example, future work could investigate routing Transformer hidden layers or the number of active attention heads according to the coarse-to-fine denoising process. In this work, however, we focus on the routing spaces that arise directly from the spatial and sequential properties of Mamba.

**W3. Actual inference latency**

**Response**

* **Summary:** While MoH significantly reduces active FLOPs and parameter footprint, current wall-clock latency is higher due to kernel launch overheads; we respectfully refer the reviewer to our response for **mhMg W1–W2** for the detailed analysis and planned CUDA optimizations.

We agree that reduced active computation does not yet translate into wall-clock acceleration in the current implementation. Under the same RTX A6000, AMP, and batch size of 8, MoH requires 87.24 ms per forward pass, compared with 61.93 ms for ZigMa. We therefore revise our efficiency claim to reduced active computation and parameter footprint rather than faster wall-clock inference.

Profiling shows that the gap is caused mainly by fragmented dynamic execution. MoH produces 3,191 kernel launches per forward pass, compared with 743 for ZigMa, while its average CUDA event duration is only 31 μs, compared with 158 μs. Routing operations such as `nonzero`, `index_select`, `index_copy_`, `scatter_`, and `topk` introduce feature gathering, branch dispatch, and output scattering.

The profiler-summed CUDA self time is 116.0 ms for MoH and 122.2 ms for ZigMa. Although this is not equivalent to wall-clock latency, it is consistent with the interpretation that the gap comes from small-kernel dispatch and memory operations rather than larger dense computation.

Based on this analysis, we are developing a lightweight dispatch-and-combine CUDA extension that groups samples sharing a route, removes host-side scalar checks, and fuses feature gathering and output combination without replacing the optimized Mamba selective-scan kernels. We are also investigating CUDA graph capture for frequent route configurations. We will release the optimized execution path, profiling scripts, and measurements in the public repository. Until these measurements are available, we explicitly report the current wall-clock limitation and restrict our claim to reduced active computation and parameter footprint.

*(For additional details on the computational efficiency profile and runtime trade-offs, please also refer to our response to **mhMg W1–W2**.)*

**W4. Routing overhead and system complexity**

**Response**

Overall routing overhead is lightweight, adding only 0.666M parameters, corresponding to 0.52% of MoH-Base. Its arithmetic cost is 0.0015 GFLOPs per sample, below 0.002% of the total 80.9 GFLOPs. Thus, the routing overhead itself is negligible relative to the performance gain. The remaining wall-clock overhead mainly arises from inefficient hardware execution rather than from route prediction. We will explicitly discuss this limitation and hardware-aware optimization as future work.

**W5. Equation clarity**

**Response**

We agree and will revise Eq. (4) to explicitly clarify that *s* denotes the hierarchy stage and *z<sub>s</sub>* denotes the latent representation at stage *s*.

**Q1. Training stability**

**Response**

Dynamic routing does not cause optimization collapse or late-stage instability. Both training loss and FID improve consistently throughout training.

*Training progression of the dynamically routed MoH model.*

| Training step | 100 | 10k | 30k | 50k | 70k | 90k | 100k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Training loss ↓ | 1.5484 | 0.7152 | 0.6608 | 0.6443 | 0.6420 | 0.6379 | 0.6356 |
| FID<sup>5k</sup> ↓ | 666.00 | 314.25 | 40.76 | 10.74 | 8.88 | 8.37 | 7.99 |

Both the training loss and FID converge smoothly. The fixed hierarchical baseline without routing and Zigma model also converges within the same 100k-step schedule. Thus, routing does not slow convergence or introduce late-stage instability.

The routers also do not collapse. SSR selects stages 32, 16, and 8 with mean ratios of 0.7762, 0.6330, and 0.5908. CMR uses all three compression choices, with mean stride-0/1/2 ratios of 0.2014/0.2158/0.5829 at resolution 16 and 0.2607/0.2858/0.4534 at resolution 8. EDR likewise uses all depth choices, with shallow/base/deep ratios of 0.1949/0.3624/0.4427 at stage 16 and 0.3240/0.3810/0.2950 at stage 8.

These results show stable optimization and non-collapsed, structured routing across stages, compression levels, and depths. We will include the complete training and routing-utilization curves in the revised manuscript.
