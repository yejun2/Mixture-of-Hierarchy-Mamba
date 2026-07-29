\concern{W2. Coordination among routers}

\response

In MoH, SSR first selects the active hierarchical stages, and its routing output $\alpha_t^{\mathrm{SSR}}$ is passed to CMR and EDR. The integrated routing controller (IRC) additionally provides a shared routing prior $p_t$:

[
a_t^{\mathrm{SSR}}
==================

\mathrm{MLP}_{\mathrm{SSR}}(c_t),
\qquad
p_t
===

\mathrm{MLP}_{\mathrm{IRC}}(c_t),
]

[
a_{t,s}^{\mathrm{CMR}}
======================

\mathrm{MLP}*{\mathrm{CMR},s}
\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr),
\qquad
a*{t,s}^{\mathrm{EDR}}
======================

\mathrm{MLP}_{\mathrm{EDR},s}
\bigl([c_t;\alpha_t^{\mathrm{SSR}};p_t]\bigr).
]

Thus, CMR and EDR are conditioned on both the selected hierarchy and a shared routing prior.

This dependency reflects the distinct roles of the routing spaces. SSR operates over the hierarchical stage space introduced to provide spatial and multi-scale inductive bias, while CMR and EDR refine the computation within the selected stages according to the denoising state. Since feature compression and Mamba depth jointly determine the computation path, their decisions should be coordinated for overall backbone optimization rather than optimized independently.

We have added the performance of CMR+EDR without IRC to Table~6 and will revise the manuscript accordingly. CMR+EDR without IRC obtains an FID of $11.03$, whereas CMR+EDR with IRC improves it to $9.51$. This result shows that simply enabling both routing spaces is insufficient and that sharing the IRC prior is important for coordinating their decisions.

We further analyzed the joint routing behavior of CMR and EDR using the corresponding 200k checkpoints under identical latent inputs and timesteps. IRC does not merely suppress high-compute routing combinations. Instead, it changes the joint routing policy from independent, compensation-like decisions to shared capacity allocation. Without IRC, CMR and EDR frequently select opposing routes, resulting in a high proportion of zero-sum routing states, with $P(\mathrm{load}=0)=39.99%$. With IRC, this decreases to $11.39%$, while both low-capacity and high-capacity co-activations become more frequent.

In particular,
$P(\mathrm{EDR{=}strong}\mid\mathrm{CMR{=}strong})$
increases from $2.79%$ without IRC to $42.82%$ with IRC. The timestep-wise analysis further shows that this joint behavior is not fixed. At $t=0.05$, IRC increases strong CMR--strong EDR co-activation from $0%$ to $48.44%$, while at intermediate timesteps ($t=0.35$--$0.55$), it increases weak CMR--weak EDR co-activation to $32.91$--$40.23%$.

These results indicate that IRC acts as a shared capacity-allocation mechanism. It allows CMR and EDR to jointly select a high-capacity computation path when stronger processing is required and a low-capacity path when reduced computation is sufficient. This timestep-dependent joint behavior is absent in the independent CMR+EDR setting, where the two routers exhibit more fixed or compensatory routing patterns. We will include this analysis in the Appendix of the revised manuscript.
