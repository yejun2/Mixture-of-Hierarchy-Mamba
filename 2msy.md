We sincerely thank the reviewer for taking the time to review our response and for engaging in further discussion with us.

**W2. EDR motivation**

**Table: Timestep-wise EDR-only routing distributions.**

| Timestep | shallow | base | deep |
|---:|---:|---:|---:|
| 0.05 | 0.0 | 33.3 | 66.7 |
| 0.15 | 0.0 | 33.3 | 66.7 |
| 0.25 | 33.3 | 66.7 | 0.0 |
| 0.35 | 33.3 | 66.7 | 0.0 |
| 0.45 | 66.7 | 33.3 | 0.0 |
| 0.55 | 100.0 | 0.0 | 0.0 |
| 0.65 | 100.0 | 0.0 | 0.0 |
| 0.75 | 100.0 | 0.0 | 0.0 |
| 0.85 | 100.0 | 0.0 | 0.0 |
| 0.95 | 100.0 | 0.0 | 0.0 |

We thank the reviewer for requesting a clearer motivation for EDR. Each Mamba block applies an input-dependent selective state transition; therefore, increasing depth repeatedly refines and propagates the hidden state (*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, COLM 2024). Prior analysis of Vision Mamba further suggests that deeper layers produce more semantic and less position-sensitive representations (*Stochastic Layer-Wise Shuffle for Improving Vision Mamba Training*, ICML 2025).

EDR uses this depth-dependent property to adapt the degree of state propagation across denoising timesteps. As shown by the EDR-only policy, the model favors deep routes at early timesteps, then progressively shifts toward base and shallow routes. This behavior aligns with coarse-to-fine denoising: stronger repeated state transitions support early structural and semantic formation, whereas shallower paths limit unnecessary propagation when refining later spatial details. Thus, EDR is not motivated solely by generic dynamic depth; it explicitly controls the amount of selective state propagation performed by Mamba over the denoising trajectory.

**W1. Recent baselines**

We additionally include GSPN (*Parallel Sequence Modeling via Generalized Spatial Propagation Network*, CVPR 2025), a recent spatial sequence modeling backbone, as a stronger SSM-based baseline on CelebA-HQ at $256 \times 256$. GSPN performs parallel spatial propagation without self-attention, making it a relevant recent comparison for evaluating spatial modeling with an SSM-style backbone.

**Table: Comparison with Transformer-, Mamba-, and SSM-based diffusion backbones on CelebA-HQ \(256 \times 256\).**

| Method | Backbone | FID$^{5k}$ ↓ |
|---|---|---:|
| U-ViT | Transformer | 14.50 |
| DiT | Transformer | 14.64 |
| ZigMa | Mamba | 14.27 |
| GSPN B/2 | SSM | 21.00 |
| GSPN L/2 | SSM | 20.71 |
| **MoH (ours)** | Mamba | **7.99** |

For a controlled comparison, GSPN B/2 was selected as a parameter-comparable model, with \(141.5\)M parameters and \(36.17\) GFLOPs, compared with \(129\)M parameters and \(80.88\) GFLOPs for MoH-Base. We further evaluate GSPN L/2, a substantially larger model with \(461.1\)M parameters and \(118.14\) GFLOPs, exceeding MoH-Base in both model size and computation. Both GSPN models were trained using the same batch setting and the same \(100\)k-step training schedule as MoH.

The GSPN comparison provides a complementary evaluation against a recent SSM-based spatial modeling approach. Despite using a parameter-comparable GSPN B/2 and a substantially larger GSPN L/2 with higher GFLOPs than MoH-Base, MoH achieves substantially better FID under the same batch setting and \(100\)k-step training schedule.
