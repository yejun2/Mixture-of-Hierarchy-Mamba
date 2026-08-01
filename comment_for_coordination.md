# NeurIPS 2026 Author Response


**W2. Coordination among routers**

**Table 1: Timestep-wise CMR routing distributions.**

| Timestep | CMR-only | | | CMR+EDR w/o IRC | | | CMR+EDR with IRC | | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **weak** | **base** | **strong** | **weak** | **base** | **strong** | **weak** | **base** | **strong** |
| 0.05 | 50.0 | 0.0 | 50.0 | 50.0 | 50.0 | 0.0 | 0.0 | 51.6 | 48.4 |
| 0.15 | 0.0 | 50.0 | 50.0 | 61.0 | 38.6 | 0.4 | 5.3 | 39.8 | 54.9 |
| 0.25 | 0.0 | 50.0 | 50.0 | 39.7 | 50.5 | 9.8 | 53.0 | 7.0 | 39.9 |
| 0.35 | 41.9 | 50.0 | 8.1 | 1.3 | 48.7 | 50.0 | 89.2 | 1.4 | 9.5 |
| 0.45 | 50.0 | 47.0 | 3.0 | 0.0 | 50.0 | 50.0 | 96.1 | 0.2 | 3.7 |
| 0.55 | 58.4 | 0.0 | 41.6 | 0.0 | 50.0 | 50.0 | 98.5 | 0.0 | 1.5 |
| 0.65 | 77.1 | 0.0 | 22.9 | 47.9 | 2.1 | 50.0 | 99.3 | 0.0 | 0.7 |
| 0.75 | 96.3 | 0.0 | 3.7 | 49.8 | 0.2 | 50.0 | 99.4 | 0.0 | 0.6 |
| 0.85 | 100.0 | 0.0 | 0.0 | 50.0 | 0.0 | 50.0 | 100.0 | 0.0 | 0.0 |
| 0.95 | 100.0 | 0.0 | 0.0 | 50.0 | 0.0 | 50.0 | 100.0 | 0.0 | 0.0 |

<br>

**Table 2: Timestep-wise EDR routing distributions.**

| Timestep | EDR-only | | | CMR+EDR w/o IRC | | | CMR+EDR with IRC | | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **weak** | **base** | **strong** | **weak** | **base** | **strong** | **weak** | **base** | **strong** |
| 0.05 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 33.3 | 1.4 | 65.4 |
| 0.15 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 34.8 | 51.1 | 14.1 |
| 0.25 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 36.3 | 62.0 | 1.7 |
| 0.35 | 100.0 | 0.0 | 0.0 | 0.0 | 66.7 | 33.3 | 34.0 | 65.7 | 0.3 |
| 0.45 | 100.0 | 0.0 | 0.0 | 0.0 | 66.7 | 33.3 | 27.9 | 71.9 | 0.2 |
| 0.55 | 66.7 | 33.3 | 0.0 | 0.0 | 100.0 | 0.0 | 22.9 | 77.0 | 0.1 |
| 0.65 | 33.3 | 66.7 | 0.0 | 33.3 | 33.3 | 33.3 | 11.3 | 88.4 | 0.3 |
| 0.75 | 33.3 | 66.7 | 0.0 | 33.3 | 33.3 | 33.3 | 0.6 | 95.2 | 4.2 |
| 0.85 | 0.0 | 33.3 | 66.7 | 100.0 | 0.0 | 0.0 | 0.0 | 99.2 | 0.8 |
| 0.95 | 0.0 | 33.3 | 66.7 | 100.0 | 0.0 | 0.0 | 0.7 | 99.2 | 0.2 |

<br>

**Table 3: Conditional EDR routing distributions given the CMR decision.**

| Timestep range | Model | CMR decision | EDR decision | Probability |
| :--- | :--- | :--- | :--- | ---: |
| **Early (0.00–0.33)** | **w/o IRC** | weak ($n=1545$, 50.3%) | weak | 0.0% |
| | | | base | 0.0% |
| | | | strong | 100.0% |
| | | base ($n=1424$, 46.4%) | weak | 0.0% |
| | | | base | 0.0% |
| | | | strong | 100.0% |
| | | strong ($n=103$, 3.4%) | weak | 0.0% |
| | | | base | 0.0% |
| | | | strong | 100.0% |
| | **with IRC** | weak ($n=597$, 19.4%) | weak | 24.8% |
| | | | base | 74.5% |
| | | | strong | 0.7% |
| | | base ($n=1008$, 32.8%) | weak | 12.4% |
| | | | base | 33.3% |
| | | | strong | 54.3% |
| | | strong ($n=1467$, 47.8%) | weak | 38.7% |
| | | | base | 14.0% |
| | | | strong | 47.3% |
| **Mid (0.34–0.66)** | **w/o IRC** | weak ($n=504$, 12.3%) | weak | 0.0% |
| | | | base | 0.0% |
| | | | strong | 100.0% |
| | | base ($n=1544$, 37.7%) | weak | 0.0% |
| | | | base | 33.2% |
| | | | strong | 66.8% |
| | | strong ($n=2048$, 50.0%) | weak | 0.0% |
| | | | base | 100.0% |
| | | | strong | 0.0% |
| | **with IRC** | weak ($n=3923$, 95.8%) | weak | 33.2% |
| | | | base | 66.5% |
| | | | strong | 0.3% |
| | | base ($n=16$, 0.4%) | weak | 12.5% |
| | | | base | 87.5% |
| | | | strong | 0.0% |
| | | strong ($n=157$, 3.8%) | weak | 99.4% |
| | | | base | 0.6% |
| | | | strong | 0.0% |
| **Late (0.67–1.00)** | **w/o IRC** | weak ($n=1534$, 49.9%) | weak | 66.8% |
| | | | base | 0.0% |
| | | | strong | 33.2% |
| | | base ($n=2$, 0.1%) | weak | 0.0% |
| | | | base | 0.0% |
| | | | strong | 100.0% |
| | | strong ($n=1536$, 50.0%) | weak | 66.7% |
| | | | base | 33.3% |
| | | | strong | 0.0% |
| | **with IRC** | weak ($n=3066$, 99.8%) | weak | 0.6% |
| | | | base | 97.0% |
| | | | strong | 2.4% |
| | | base ($n=0$, 0.0%) | weak | -- |
| | | | base | -- |
| | | | strong | -- |
| | | strong ($n=6$, 0.2%) | weak | 16.7% |
| | | | base | 16.7% |
| | | | strong | 66.7% |

<br>

CMR and EDR provide greater routing capacity when used together, but this capacity is beneficial only when their decisions are coordinated. Without IRC, the two routers optimize separate local policies under the same generation loss. This disrupts the coarse-to-fine CMR behavior and causes EDR to adopt saturated decisions that are largely insensitive to the selected CMR route.

CMR and EDR have asymmetric roles. CMR determines the spatial compression and feature-mixing configuration presented to the Mamba blocks, while EDR, when jointly applied with CMR, selects the computation depth appropriate for that configuration. With IRC, CMR preserves the overall coarse-to-fine trend of the effective CMR-only policy, while EDR adapts its weak, base, and strong choices to both the timestep and the CMR-selected spatial configuration.

The conditional statistics directly show this coordination. During the early interval, EDR without IRC selects the strong route with 100% probability regardless of whether CMR selects a weak, base, or strong route, indicating that its depth policy is effectively decoupled from the spatial configuration. With IRC, the EDR distribution changes substantially according to the CMR decision: a weak CMR route is followed by a base EDR route in 74.5% of cases, whereas a base CMR route is followed by a strong EDR route in 54.3% of cases. This dependency also changes across the denoising process. In the middle interval, a strong CMR route is paired with a weak EDR route in 99.4% of cases, while in the late interval, a weak CMR route is paired with a base EDR route in 97.0% of cases. These results show that IRC enables EDR to adjust its computation depth according to both the selected spatial configuration and the denoising stage. Notably, in the late interval with IRC, CMR selects weak in 3066 cases, strong only 6 times, and never selects base, closely matching the CMR-only policy and exhibiting the intended coarse-to-fine pattern.

Thus, IRC does not force the routers to make identical decisions. It preserves the useful spatial-routing structure of CMR while adapting EDR to the resulting compression and mixing context. This coordinated behavior improves the FID from 11.00 to 9.51.

---

**W4. Architectural routing versus token/layer routing**

**Response.**

We thank the reviewer for this important clarification. We agree that orthogonality alone does not establish why architectural routing is preferable in our setting, and we do not claim that it is universally superior to token- or layer-level routing. Rather, architectural routing is better matched to the specific granularity and operations that must be adapted in our hierarchical Mamba diffusion backbone.

First, the diffusion timestep is a global variable that characterizes the denoising state of the entire image. The transition from coarse structural formation at early timesteps to fine-detail refinement at later timesteps is therefore closer to a network-level capacity-allocation problem than to independently assigning experts to individual tokens. This interpretation is consistent with *Denoising Task Routing for Diffusion Models* (DTR, ICLR 2024), which treats timesteps as distinct denoising tasks and assigns different channel pathways and capacities to them. Likewise, *Dynamic Diffusion Transformer* (DyDiT, ICLR 2025) explicitly separates timestep-wise global width adaptation from spatial-wise token routing, supporting the view that global denoising variation and local token redundancy require different routing granularities.

Second, the routed operations in MoH are architectural rather than token-wise. Standard Mamba performs sequential scanning and state propagation, but it does not explicitly provide a multi-scale spatial hierarchy or a mechanism for controlling spatial aggregation across resolutions. We therefore organize the Mamba scanning space hierarchically. SSR selects the active spatial stages, while CMR controls how strongly features are spatially mixed and compressed before Mamba scanning. In this sense, CMR determines the spatial aggregation scale presented to Mamba rather than assigning individual tokens to experts. Under this selected spatial configuration, EDR controls the amount of stage-wise state propagation. Together, these routers modify the spatial organization and propagation path of the backbone itself.

Token routing may still refine computation among individual tokens within an already selected path. However, it does not directly determine which resolutions are active, how much cross-scale spatial aggregation is applied, or how deeply Mamba propagates information at each stage. Architectural routing is therefore preferable here because the target of adaptation is the global spatial hierarchy and state-propagation structure of Mamba computation. We will revise Sec. 2 to make this task- and architecture-specific motivation clearer and remove any wording that could be interpreted as claiming universal superiority.
