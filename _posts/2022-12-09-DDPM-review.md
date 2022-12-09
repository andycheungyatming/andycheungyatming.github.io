---
layout:       post
title:        "Review: Denoising Diffusion Probabilistic Models (DDPM)"
author:       "Allan"
header-style: text
catalog:      true
tags:
    - AI
    - Review
    - Diffusion
    - Generation
---
# What are Diffusion Models?
Denoising Diffusion Probabilistic Models (DDPM) is introduced in [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239). The maths background is discussed [(Sohl-Dickstein et al., 2015)](https://arxiv.org/abs/1503.03585). The essential idea is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process which is fixed.

# Some maths on DDPM
![figure1](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)
>The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise.

we can let a image smaple(from real data distribution)
$$ \mathbf{x}_0 \sim q(\mathbf{x}) $$

Usually we will also define steps
$$T$$
, where step size is controlled by 
$$ \{\beta_t \in (0, 1)\}_{t=1}^T $$

With steps 
$$T$$
, we can produce a sequence of noisy samples

$$ x_1, x_2, ..., x_T = z$$

However, it is difficult to rebuild the image from $$x_t$$ to $$x_0$$ directly, we need the model to learn the rebuilt process piece by piece. 
i.e.

$$ x_T \rarr x_{T-1} = u(x_T) \rarr x_{T-2} = u(x_{T-1}) \rarr ... \rarr x_1 = u(x_2) \rarr x_0 = u(x_1) $$

## Destruction (Forward process)
I'd like to use destruction instead of forward process. Basically we want to make a image (with pattern) to a pure gaussian noise by putting more gaussian noise recursively (with a fixed number of steps). 
Let 
$$ \alpha_t + \beta_t = 1 $$
and 
$$ \bar{\alpha}_t = \prod^t_{i=1}\alpha_i$$
, we have 

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

