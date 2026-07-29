To further analyze the effect of the integrated routing controller (IRC), we compared the routing behavior of CMR+EDR with and without IRC using the 200k checkpoints. We used the same latent inputs and the same diffusion timestep grid for both models, and performed inference-only forward probing without sampling, FID evaluation, or parameter updates. Each CMR and EDR decision was mapped onto an ordinal routing-strength axis, where weak/base/strong correspond to -1/0/+1. We then measured the joint routing state between CMR and EDR at each diffusion timestep. This allows us to evaluate whether the two routing spaces behave independently or whether they form a shared timestep-dependent routing policy.

| Diffusion timestep | Model | CMR strong rate | P(CMR strong, EDR strong) | P(CMR weak, EDR weak) | P(extreme joint state) | P(load = 0) | E\|load\| |
|---:|---|---:|---:|---:|---:|---:|---:|
| **0.05** | w/o IRC | 0.00% | 0.00% | 0.00% | 0.00% | **50.00%** | 0.500 |
| **0.05** | **with IRC** | **48.44%** | **48.44%** | 0.00% | **48.44%** | 2.05% | **1.464** |
| **0.15** | w/o IRC | 0.29% | 0.29% | 0.00% | 0.29% | **61.04%** | 0.393 |
| **0.15** | **with IRC** | **54.88%** | **17.29%** | 1.17% | **18.46%** | 46.00% | **0.725** |
| **0.35** | w/o IRC | **50.00%** | 0.00% | 0.00% | 0.00% | 1.27% | 0.987 |
| **0.35** | **with IRC** | 9.47% | 0.00% | **40.23%** | **40.23%** | **11.04%** | **1.292** |
| **0.45** | w/o IRC | **50.00%** | 0.00% | 0.00% | 0.00% | 0.00% | 1.000 |
| **0.45** | **with IRC** | 3.71% | 0.00% | **37.79%** | **37.79%** | **4.20%** | **1.336** |
| **0.55** | w/o IRC | **50.00%** | 0.00% | 0.00% | 0.00% | **50.00%** | 0.500 |
| **0.55** | **with IRC** | 1.46% | 0.00% | **32.91%** | **32.91%** | 1.56% | **1.313** |
| **0.85** | w/o IRC | **50.00%** | 0.00% | **50.00%** | **50.00%** | **50.00%** | 1.000 |
| **0.85** | **with IRC** | 0.00% | 0.00% | 0.00% | 0.00% | 1.17% | 0.988 |

The results show that IRC does not simply suppress strong routes or force all routers toward low-compute choices. Instead, IRC produces a more timestep-dependent joint routing policy. At early diffusion timesteps, where the input is closer to noise and semantic/global processing is more important, IRC strongly co-activates CMR and EDR. For example, at timestep 0.05, the probability of the joint state CMR=strong and EDR=strong increases from 0.00% without IRC to 48.44% with IRC. This indicates that the integrated controller identifies early denoising states as requiring jointly high-capacity routing, rather than allowing CMR and EDR to make independent local decisions.

At intermediate timesteps, the behavior changes substantially. Around timesteps 0.35-0.55, IRC shifts toward low-capacity co-activation, with P(CMR=weak, EDR=weak) reaching 40.23%, 37.79%, and 32.91%, respectively. This suggests that IRC is not merely increasing routing strength; rather, it allocates joint routing capacity depending on the diffusion timestep. In contrast, without IRC, CMR often stays in a fixed strong-routing pattern over a wide timestep range, while EDR decisions are made independently. This leads to many zero-sum or compensatory states, where one router selects a strong route while the other selects a weak/base route, but without an explicit shared decision mechanism.

Overall, this routing analysis supports the role of IRC as a shared capacity-allocation mechanism. The key difference is not that IRC always reduces routing cost, but that it changes the structure of the joint routing distribution. Without IRC, CMR and EDR behave more independently and often produce fixed or compensatory routing combinations. With IRC, their decisions become jointly responsive to the same input and diffusion timestep, producing high-capacity co-activation in early denoising and low-capacity co-activation in intermediate denoising. This provides direct evidence that IRC coordinates the routing behavior of CMR and EDR rather than simply adding another independent router.
