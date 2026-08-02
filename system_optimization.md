**System-level optimization.**

We further optimized the adaptive-$k$ MoH-Hybrid inference path at the implementation level. In the original PyTorch implementation, routing decisions were executed at multiple routing boundaries, causing fragmented GPU execution through repeated route-dependent gathering, branch execution, and layout restoration.

Our routing formulation allows these decisions to be consolidated. SSR first determines the active hierarchical stages, and its routing output is shared with CMR and EDR. The integrated routing controller additionally provides a shared routing prior $p_t$:

$$a_t^{\mathrm{SSR}}=\mathrm{MLP}_{\mathrm{SSR}}(c_t), \qquad p_t=\mathrm{MLP}_{\mathrm{IRC}}(c_t),$$

$$a_{t,s}^{\mathrm{CMR}} = \mathrm{MLP}_{\mathrm{CMR},s}\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr), \qquad a_{t,s}^{\mathrm{EDR}} = \mathrm{MLP}_{\mathrm{EDR},s}\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr).$$

Thus, CMR and EDR are conditioned on both the selected hierarchy and the shared routing prior. This allows the encoder-side dynamic computation plan to be determined in a single consolidated routing step, rather than interleaving routing decisions with each intermediate feature transformation.

For the decoder, we similarly removed unnecessary dynamic dispatch from the batch-size-1 inference path. The original dynamic bottleneck decoder grouped samples according to the selected starting resolution and restored the batch layout after route-specific decoding. This grouping is unnecessary at batch size 1, so we directly execute the selected decoder route. Since the possible decoder starting resolutions are finite, we additionally capture route-specific CUDA graphs for the decoder and final-head computation:

[ `start@8`, `start@16`, `start@32`. ]

At inference time, the encoder and routing controller remain adaptive. Once the decoder starting route is selected, the corresponding CUDA graph is replayed. This preserves the learned adaptive routing behavior while reducing kernel-launch overhead and dynamic-dispatch fragmentation.

With these PyTorch-level execution restructuring and CUDA graph replay optimizations, the batch-size-1 latency of adaptive-$k$ MoH-Hybrid on a single NVIDIA RTX A6000 decreases from 49.71 ms to 36.65 ms, corresponding to a 26.3% wall-clock improvement. This result is achieved without changing the learned model, disabling dynamic routing, or implementing a custom CUDA kernel. We are also implementing custom CUDA kernels to further reduce routing and dynamic-dispatch overhead.
