**W2. Coordination among routers**

**Response**

To understand why combining CMR and EDR without IRC does not improve performance over the single-router variants, while CMR+EDR with IRC does, we analyze their routing behavior. We use the same latent inputs and the same diffusion timestep grid for both models, and perform inference-only forward probing without sampling, FID evaluation, or parameter updates. Each CMR and EDR decision is mapped onto an ordinal routing-strength axis: weak/base/strong = -1/0/+1. We then define the joint routing load as `load = CMR_strength + EDR_strength`. Under this definition, `P(extreme joint state)` measures how often the two routers jointly select the same non-base extreme state, i.e., `|load| = 2` such as weak+weak or strong+strong; `P(load = 0)` measures zero-sum routing states such as strong+weak, weak+strong, or base+base; and `E|load|` measures the average magnitude of joint routing strength.

| Diffusion timestep | Model | P(CMR strong, EDR strong) | P(CMR weak, EDR weak) | **P(extreme joint state)** | **P(load = 0)** | **E\|load\|** |
|---:|---|---:|---:|---:|---:|---:|
| **0.05** | w/o IRC | 0.00% | 0.00% | 0.00% | **50.00%** | 0.500 |
| **0.05** | **with IRC** | **48.44%** | 0.00% | **48.44%** | 2.05% | **1.464** |
| **0.15** | w/o IRC | 0.29% | 0.00% | 0.29% | **61.04%** | 0.393 |
| **0.15** | **with IRC** | **17.29%** | 1.17% | **18.46%** | 46.00% | **0.725** |
| **0.35** | w/o IRC | 0.00% | 0.00% | 0.00% | 1.27% | 0.987 |
| **0.35** | **with IRC** | 0.00% | **40.23%** | **40.23%** | **11.04%** | **1.292** |
| **0.45** | w/o IRC | 0.00% | 0.00% | 0.00% | 0.00% | 1.000 |
| **0.45** | **with IRC** | 0.00% | **37.79%** | **37.79%** | **4.20%** | **1.336** |
| **0.55** | w/o IRC | 0.00% | 0.00% | 0.00% | **50.00%** | 0.500 |
| **0.55** | **with IRC** | 0.00% | **32.91%** | **32.91%** | 1.56% | **1.313** |
| **0.85** | w/o IRC | 0.00% | **50.00%** | **50.00%** | **50.00%** | 1.000 |
| **0.85** | **with IRC** | 0.00% | 0.00% | 0.00% | 1.17% | 0.988 |

The results suggest that the benefit of IRC is not simply from enabling both CMR and EDR, but from coordinating their decisions. Without IRC, CMR and EDR are both active, but they make routing decisions independently. This often produces zero-sum or inconsistent routing states. For example, at timestep 0.05 and 0.15, `P(load = 0)` is 50.00% and 61.04% without IRC, while CMR almost never selects the strong route. This means that the combined router system is not behaving like a coherent joint policy; instead, the two routing spaces frequently cancel or decouple from each other. This provides a plausible explanation for why simply combining CMR and EDR without IRC can perform slightly worse than using only CMR or only EDR: the additional routing degree of freedom increases system complexity, but the routers are not jointly optimized to make compatible decisions.

With IRC, the joint routing distribution becomes much more timestep-dependent. At early diffusion timesteps, IRC assigns high joint capacity: at timestep 0.05, `P(CMR strong, EDR strong)` increases from 0.00% without IRC to 48.44% with IRC, and `E|load|` increases from 0.500 to 1.464. At intermediate timesteps, IRC shifts toward low joint capacity: `P(CMR weak, EDR weak)` reaches 40.23%, 37.79%, and 32.91% at timesteps 0.35, 0.45, and 0.55, respectively. Thus, IRC does not merely push routers toward stronger or weaker choices globally. Instead, it makes CMR and EDR jointly responsive to the denoising stage, allocating more capacity when both compression and depth processing are useful, and reducing capacity when a lighter joint path is sufficient.

This analysis supports the interpretation that shared prior from IRC is necessary for making multiple routers work together. CMR+EDR without IRC introduces two adaptive routing spaces, but their independent decisions can produce uncoordinated or zero-sum combinations, which may explain why the performance does not improve over single-router variants. In contrast, CMR+EDR with IRC converts the two routers into a coordinated routing system. The improvement is therefore not merely due to adding more routing modules, but due to aligning CMR and EDR through a shared controller that produces timestep-dependent joint capacity allocation.
