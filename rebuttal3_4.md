We thank the reviewer for recognizing the substantial performance gains and the alignment between timestep-conditioned routing and the coarse-to-fine nature of diffusion.

**W1. Recent baselines**

**Response**

We agree that recent Mamba-based diffusion models should be better covered. We therefore add U-Shape Mamba, the most directly related recent Self-attention-free Mamba baseline evaluated on MS-COCO at $256 \times 256$.

*Conditional image generation on MS-COCO at $256 \times 256$.*

| Method | Backbone | FID$^{5k}$ $\downarrow$ |
| :--- | :---: | :---: |
| VisionMamba | Mamba | $60.20$ |
| ZigMa | Mamba | $41.80$ |
| U-Shape Mamba | Mamba | $39.10$ |
| MoH (ours) | Mamba | $\mathbf{20.00}$ |

U-Shape Mamba forms a U-shaped hierarchy by reducing the sequence after Mamba processing and reconstructing it in the decoder through upsampling and skip connections. In contrast, MoH treats each spatial resolution as a distinct Mamba scanning space, so the scanning process itself is performed hierarchically across multiple spatial scales.

We will also include hybrid models such as Dimba and DiMSUM as broader references. These methods combine Mamba with Transformer or self-attention modules to maximize overall backbone performance, whereas MoH focuses on improving the Mamba architecture itself without self-attention. Thus, Self-attention-free Mamba models provide the most direct controlled comparison, while hybrid models provide complementary performance context.
