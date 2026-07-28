# Response to the Meta-Review

We thank the Area Chair for consolidating the reviews and for recognizing the novelty and empirical potential of timestep-conditioned architectural routing. We address each consolidated concern below.

**1. Baselines and comparisons**

**Response**

We add U-Shape Mamba, a recent self-attention-free Mamba diffusion model on MS-COCO 256 × 256. It reports FID 39.10, compared with 41.80 for ZigMa and 20.00 for MoH. U-Shape Mamba forms its hierarchy by downsampling Mamba-processed sequences and reconstructing them through upsampling and skip connections. In contrast, MoH treats each spatial resolution as a distinct Mamba scanning space, so Mamba scanning itself is performed hierarchically across spatial scales.

We will also discuss Dimba and DiMSUM as broader hybrid references. Since they include Transformer or self-attention modules, we separate them from controlled self-attention-free Mamba comparisons.

Coordination among routers is addressed through the revised formulation and incremental ablation below: SSR provides the upper-level stage decision to the stage-wise CMR and EDR routers, while Table 6 measures their individual and joint contributions.

**2. Inference efficiency**

**Response**

We agree that reduced active computation does not yet yield faster wall-clock inference. Under the same RTX A6000, AMP, and batch size 8, MoH requires 87.24 ms per denoiser evaluation, compared with 61.93 ms for ZigMa. We therefore revise the efficiency claim to reduced active computation and parameter footprint rather than faster wall-clock execution.

MoH requires 80.88 GFLOPs, compared with 97.52 GFLOPs for ZigMa, but produces 3,191 kernel launches per forward pass versus 743. Its average CUDA event duration is 31 μs versus 158 μs, indicating fragmented small-kernel execution and gather/scatter overhead.

The routing controller itself adds only 0.666M parameters (0.52% of MoH-Base) and 0.0015 GFLOPs (<0.002% of the total). Thus, the latency gap primarily arises from hardware-inefficient dynamic execution rather than route prediction. We will report this limitation explicitly and discuss grouped dispatch and hardware-aware execution as future work.

**3. Motivation and scope of architectural routing**

**Response**

Our goal is to design a self-attention-free Mamba backbone better suited to vision. CNNs explicitly encode locality and spatial hierarchy, while Transformers model broad spatial relations through direct token interactions. Mamba originates from one-dimensional state propagation and lacks an explicit multi-scale spatial inductive bias.

We therefore introduce spatially hierarchical Mamba scanning spaces. CMR adapts spatial mixing and compression across denoising timesteps. EDR is motivated by the interaction between hierarchy and Mamba propagation: higher-resolution stages preserve local detail, lower-resolution stages favor semantic aggregation, and Mamba depth controls the degree of information integration. These choices allow the backbone capacity to follow the coarse-to-fine denoising process.

Architectural routing is not intended to replace or outperform token- or layer-level routing. Token routing dispatches individual tokens to expert subnetworks based on input features, whereas MoH routes over spatial stages, feature transformation, and Mamba depth. The two can coexist. For example, after EDR selects the stage-wise depth, token-dependent routing could be added within the selected computation path. Extending this principle to other backbones, such as routing Transformer depth or active attention heads, is promising future work.

**4. Hierarchy, routing, and coordination**

**Response**

We agree that the current Table 6 does not clearly separate the contributions of hierarchy and routing. The fixed hierarchical Mamba backbone improves FID from 14.27 to 11.93, a 2.34 improvement. Starting from this backbone, CMR and EDR improve FID to 10.60 and 10.81, respectively. Their joint configuration reaches 9.51, and adaptive-*k* SSR further improves it to 9.28. Thus, timestep-conditioned routing provides an additional 2.65 FID improvement over the fixed hierarchy.

We will revise Table 6 as follows:

| Model | SSR | CMR | EDR | IRC | FID<sup>5k</sup> ↓ | sFID<sup>5k</sup> ↓ | FDD<sup>5k</sup> ↓ | IS<sup>5k</sup> ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MoH | top-3 | -- | -- | -- | 11.93 | 10.56 | 3.45 | 3.04 |
| MoH | top-3 | ✓ | -- | -- | 10.60 | 9.16 | 2.70 | 2.95 |
| MoH | top-3 | -- | ✓ | -- | 10.81 | 8.93 | 2.93 | 2.80 |
| MoH | top-3 | ✓ | ✓ | -- | 11.00 | 9.22 | 3.14 | 2.86 |
| MoH | top-3 | ✓ | ✓ | ✓ | 9.51 | 8.39 | 2.46 | 2.91 |
| MoH | adaptive-k | ✓ | ✓ | ✓ | **9.28** | 8.50 | 2.74 | 2.95 |

We also clarify the coordination mechanism.

In MoH, SSR first selects the active hierarchical stages, and its routing output $\alpha_t^{\mathrm{SSR}}$ is passed to CMR and EDR. The integrated routing controller (IRC) additionally provides a shared routing prior $p_t$:

$$
a_t^{\mathrm{SSR}}=\mathrm{MLP}_{\mathrm{SSR}}(c_t), \qquad p_t=\mathrm{MLP}_{\mathrm{IRC}}(c_t),
$$

$$
a_{t,s}^{\mathrm{CMR}} = \mathrm{MLP}_{\mathrm{CMR},s} \bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr), \qquad a_{t,s}^{\mathrm{EDR}} = \mathrm{MLP}_{\mathrm{EDR},s} \bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr).
$$

Thus, CMR and EDR are conditioned on both the selected hierarchy and a shared routing prior.

We have added the performance of CMR+EDR without IRC to Table 6 and will revise the manuscript accordingly. CMR+EDR without IRC obtains an FID of 11.03, whereas CMR+EDR with IRC improves it to 9.51. This variant obtains an FID of 11.03, whereas the shared IRC improves it to 9.51. The result directly shows that simply enabling both routers is insufficient, while coordinating their decisions through a shared prior substantially improves performance. Since feature compression and Mamba depth jointly determine the computation path, they should be coordinated rather than optimized independently.

We are currently analyzing the differences in routing behavior between CMR + EDR with IRC and CMR + EDR without IRC. We will share these findings in a comment during the discussion period and include the complete analysis in the Appendix of the revised manuscript.

**5. System complexity and training stability**

**Response**

The routing controller adds only 0.666M parameters and 0.0015 GFLOPs, which is negligible relative to the model size, computation, and observed FID gain.

Dynamic routing also remains stable throughout training. Loss decreases from 1.5484 at step 100 to 0.6356 at 100k, while FID improves to 10.74 at 50k, 8.37 at 90k, and 7.99 at 100k, without late-stage divergence.

We observe no routing collapse. SSR uses stages 32/16/8 with mean selection ratios 0.7762/0.6330/0.5908, and CMR and EDR use all available choices across stages. We will add the complete training and routing-utilization curves.

**6. Presentation and reviewer-specific clarifications**

**Response**

We will define *s* as the hierarchy stage and *z<sub>s</sub>* as its latent representation in Eq. (4), explicitly show the dependence of CMR and EDR on $\alpha_t^{\mathrm{SSR}}$, and correct $K_c$ to $K_d$ in Eq. (11).
