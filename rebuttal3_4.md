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
