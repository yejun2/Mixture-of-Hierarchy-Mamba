**Response to Reviewer UfBF**

We thank the reviewer for these insightful comments. We agree on the importance of clarifying the distinct roles of our backbone versus dynamic routing, formalizing router coordination, and emphasizing that architectural routing complements—rather than competes with—token- or layer-level routing.

**W1. Whether the architectural-routing viewpoint drives the gain**

**Response**

We thank the reviewer for this important observation. We agree that the respective roles of the hierarchical backbone and timestep-conditioned routing should be distinguished more clearly.

Our contribution consists of two distinct but complementary axes.

**First, we introduce spatial hierarchy directly into the Mamba scanning process.** Vision Mamba has relatively weak spatial and multi-scale inductive biases because image features are processed primarily as flattened sequences. Rather than simply attaching a conventional hierarchical backbone to Mamba, **we introduce Mamba scanning itself across multiple spatial resolutions and recurrently integrate the resulting multi-scale contexts.** This provides Mamba with an explicit spatial representation structure that is important for visual generation. Quantitatively, this hierarchical Mamba backbone improves FID from 14.27 for the flat Mamba baseline to 11.93.

**Second, we adapt both the spatial hierarchy and Mamba processing depth to the coarse-to-fine denoising pattern of diffusion.** The relevance of each spatial resolution changes across timesteps: coarse representations are more important during global structure formation, while fine-resolution features become increasingly important during detail refinement. Accordingly, SSR selects the active spatial stages at each timestep. EDR further adjusts Mamba depth, based on the observation that deeper recurrent propagation produces more aggregated representations, whereas shallower processing better preserves local details. CMR complements this by adapting feature mixing and compression before scanning.

Starting from the same fixed hierarchical backbone, CMR and EDR improve FID from 11.93 to 10.60 and 10.81, respectively. Jointly enabling them improves FID to 9.51, and adaptive stage selection further improves it to 9.28. Thus, the hierarchy accounts for a 2.34 FID improvement over the flat Mamba baseline, while timestep-conditioned adaptation provides an additional 2.65 FID improvement over the fixed hierarchical model.

We agree that the current presentation of Table 6 does not make this distinction sufficiently explicit. **We will revise Table 6 as shown below to make this distinction explicit.**

| Model | SSR | CMR | EDR | IRC | FID<sup>5k</sup> ↓ | sFID<sup>5k</sup> ↓ | FDD<sup>5k</sup> ↓ | IS<sup>5k</sup> ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MoH | top-3 | - | - | - | 11.93 | 10.56 | 3.45 | 3.04 |
| MoH | top-3 | ✓ | - | - | 10.60 | 9.16 | 2.70 | 2.95 |
| MoH | top-3 | - | ✓ | - | 10.81 | 8.93 | 2.93 | 2.80 |
| MoH | top-3 | ✓ | ✓ | - | 11.00 | 9.22 | 3.14 | 2.86 |
| MoH | top-3 | ✓ | ✓ | ✓ | 9.51 | 8.39 | 2.46 | 2.91 |
| MoH | adaptive-k | ✓ | ✓ | ✓ | **9.28** | 8.50 | 2.74 | 2.95 |

**W2. Coordination among routers**

**Response**

In MoH, SSR first selects the active hierarchical stages, and its routing output $$\alpha_t^{\mathrm{SSR}}$$ is passed to CMR and EDR. **The integrated routing controller (IRC) additionally provides a shared routing prior $$p_t$$:**

$$
a_t^{\mathrm{SSR}}=\mathrm{MLP}_{\mathrm{SSR}}(c_t),
\qquad
p_t=\mathrm{MLP}_{\mathrm{IRC}}(c_t)
$$

$$
a_{t,s}^{\mathrm{CMR}} = \mathrm{MLP}_{\mathrm{CMR},s}\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr), \qquad a_{t,s}^{\mathrm{EDR}} = \mathrm{MLP}_{\mathrm{EDR},s}\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr)
$$

Thus, CMR and EDR are conditioned on both the selected hierarchy and a shared routing prior.

This dependency reflects the distinct roles of the routing spaces. SSR operates over the hierarchical stage space introduced to provide spatial and multi-scale inductive bias, while CMR and EDR refine the computation within the selected stages according to the denoising state. Since feature compression and Mamba depth jointly determine the computation path, they should be coordinated rather than optimized independently.

**We have added the performance of CMR+EDR without IRC to Table~6 and will revise the manuscript accordingly. CMR+EDR without IRC obtains an FID of $11.03$, whereas CMR+EDR with IRC improves it to $9.51$.** The result directly shows that simply enabling both routers is insufficient, while coordinating their decisions through a shared prior substantially improves performance.

We are currently analyzing the differences in routing behavior between CMR + EDR with IRC and CMR + EDR without IRC. We will share these findings in a comment during the discussion period and include the complete analysis in the Appendix of the revised manuscript.

**W3. Efficiency claim**

**Response**

While MoH reduces active FLOPs and parameter footprint, current wall-clock latency is higher due to kernel launch overheads; we respectfully refer the reviewer to our response for **mhMg Q1–Q2** for the detailed analysis and planned CUDA optimizations.

**W4. Architectural routing versus token/layer routing**

**Response**

Token-level routing typically dispatches tokens to different expert sub-networks based on input features. In contrast, MoH adjusts the internal capacity of the Mamba backbone across diffusion timesteps by selecting spatial stages, feature mixing and compression, and Mamba propagation depth. The goal is to adapt the architecture to the coarse-to-fine denoising process in a manner consistent with the properties of Mamba.

Because these mechanisms operate at different levels, token-level routing could be incorporated into MoH as an additional routing space. For example, after EDR selects the stage-wise Mamba depth, a token-dependent prior could further route individual tokens within the selected computation path. This illustrates that token routing can coexist with, rather than replace, architectural routing. We view such a combination as a future work and will revise Sec. 4.4 to clarify this distinction.

**Minor: Typo in Eq. (11)**

**Response**

We thank the reviewer for identifying this typo. $K_c$ in Eq. (11) should be $K_d$, and we will correct it.
