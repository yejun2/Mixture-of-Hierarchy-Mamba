We thank the reviewer for recognizing the novelty of architectural-level routing and for noting that the learned routing behavior autonomously follows a coarse-to-fine pattern.

W1. Whether the architectural-routing viewpoint drives the gain

We thank the reviewer for this important observation. We agree that the respective roles of the hierarchical backbone and timestep-conditioned routing should be distinguished more clearly.

Our contribution consists of two distinct but complementary axes.

First, we introduce spatial hierarchy directly into the Mamba scanning process. Vision Mamba has relatively weak spatial and multi-scale inductive biases because image features are processed primarily as flattened sequences. Rather than simply attaching a conventional hierarchical backbone to Mamba, we perform Mamba scanning itself across multiple spatial resolutions and recurrently integrate the resulting multi-scale contexts. This provides Mamba with an explicit spatial representation structure that is important for visual generation. Quantitatively, this hierarchical Mamba backbone improves FID from 14.27 for the flat Mamba baseline to 11.93.

Second, we adapt both the spatial hierarchy and Mamba processing depth to the coarse-to-fine denoising pattern of diffusion. The relevance of each spatial resolution changes across timesteps: coarse representations are more important during global structure formation, while fine-resolution features become increasingly important during detail refinement. Accordingly, SSR selects the active spatial stages at each timestep. EDR further adjusts Mamba depth, based on the observation that deeper recurrent propagation produces more aggregated representations, whereas shallower processing better preserves local details. CMR complements this by adapting feature mixing and compression before scanning.

Starting from the same fixed hierarchical backbone, CMR and EDR improve FID from 11.93 to 10.60 and 10.81, respectively. Jointly enabling them improves FID to 9.51, and adaptive stage selection further improves it to 9.28. Thus, the hierarchy accounts for a 2.34 FID improvement over the flat Mamba baseline, while timestep-conditioned adaptation provides an additional 2.65 FID improvement over the fixed hierarchical model.

We agree that the current presentation of Table 6 does not make this distinction sufficiently explicit. We will revise Table 6 as shown below to make this distinction explicit.

| Model | SSR | CMR | EDR | IRC | FID<sup>5k</sup> ↓ | sFID<sup>5k</sup> ↓ | FDD<sup>5k</sup> ↓ | IS<sup>5k</sup> ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MoH | top-3 | - | - | - | 11.93 | 10.56 | 3.45 | 3.04 |
| MoH | top-3 | ✓ | - | - | 10.60 | 9.16 | 2.70 | 2.95 |
| MoH | top-3 | - | ✓ | - | 10.81 | 8.93 | 2.93 | 2.80 |
| MoH | top-3 | ✓ | ✓ | - | 11.00 | 9.22 | 3.14 | 2.86 |
| MoH | top-3 | ✓ | ✓ | ✓ | 9.51 | 8.39 | 2.46 | 2.91 |
| MoH | adaptive-k | ✓ | ✓ | ✓ | **9.28** | 8.50 | 2.74 | 2.95 |

W2. Coordination among routers

We thank the reviewer for pointing out that the coordination among the routers was not sufficiently formalized or verified.

In MoH, SSR first selects the active hierarchical stages, and its routing output $$\alpha_t^{\mathrm{SSR}}$$ is passed to CMR and EDR. The integrated routing controller (IRC) additionally provides a shared routing prior $$p_t$$:

$$
a_t^{\mathrm{SSR}}=\mathrm{MLP}_{\mathrm{SSR}}(c_t),
\qquad
p_t=\mathrm{MLP}_{\mathrm{IRC}}(c_t)
$$

$$
a_{t,s}^{\mathrm{CMR}}
=
\mathrm{MLP}_{\mathrm{CMR},s}
\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr),
\qquad
a_{t,s}^{\mathrm{EDR}}
=
\mathrm{MLP}_{\mathrm{EDR},s}
\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr)
$$

Thus, CMR and EDR are conditioned on both the selected hierarchy and a shared routing prior.

This dependency reflects the distinct roles of the routing spaces. SSR operates over the hierarchical stage space introduced to provide spatial and multi-scale inductive bias, while CMR and EDR refine the computation within the selected stages according to the denoising state. Since feature compression and Mamba depth jointly determine the computation path, they should be coordinated rather than optimized independently.

We have added the performance of CMR+EDR without IRC to Table~6 and will revise the manuscript accordingly. CMR+EDR without IRC obtains an FID of $11.03$, whereas CMR+EDR with IRC improves it to $9.51$.
This variant obtains an FID of $11.03$, whereas the shared IRC improves it to $9.51$. The result directly shows that simply enabling both routers is insufficient, while coordinating their decisions through a shared prior substantially improves performance.

W3. Efficiency claim

We agree that reduced active computation does not yet translate into wall-clock acceleration in the current implementation. Under the same RTX A6000, AMP, and batch size of $8$, MoH requires $87.24$ ms per forward pass, compared with $61.93$ ms for ZigMa. We therefore revise our efficiency claim to reduced active computation and parameter footprint rather than faster wall-clock inference.

Profiling shows that the gap is caused mainly by fragmented dynamic execution. MoH produces $$3{,}191$$ kernel launches per forward pass, compared with $$743$$ for ZigMa, while its average CUDA event duration is only $$31\,\mu\mathrm{s}$$, compared with $$158\,\mu\mathrm{s}$$. Routing operations such as \verb|nonzero|, \verb|index_select|, \verb|index_copy_|, \verb|scatter_|, and \verb|topk| introduce feature gathering, branch dispatch, and output scattering.

The profiler-summed CUDA self time is $116.0$ ms for MoH and $122.2$ ms for ZigMa. Although this is not equivalent to wall-clock latency, it is consistent with the interpretation that the gap comes from small-kernel dispatch and memory operations rather than larger dense computation.

Based on this analysis, we are developing a lightweight dispatch-and-combine CUDA extension that groups samples sharing a route, removes host-side scalar checks, and fuses feature gathering and output combination without replacing the optimized Mamba selective-scan kernels. We are also investigating CUDA graph capture for frequent route configurations. We will release the optimized execution path, profiling scripts, and measurements in the public repository. Until these measurements are available, we explicitly report the current wall-clock limitation and restrict our claim to reduced active computation and parameter footprint.
