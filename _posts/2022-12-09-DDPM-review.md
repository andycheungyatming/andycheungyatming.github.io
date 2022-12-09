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
usually we will also define steps **T**
The step size is controlled by 
$$ \{\beta_t \in (0, 1)\}_{t=1}^T $$
With steps **T**, we can produce a sequence of noisy samples 
$$ x_1, x_2, ..., x_T $$

However, it is difficult to rebuild the image from $$x_t$$ to $$x_0$$ directly, we need the model to learn the rebuilt process piece by piece. 

## Destruction (Forward process)
I'd like to use destruction instead of forward process. Basically we want to make a image (with pattern) to a pure gaussian noise by putting more gaussian noise recursively (with a fixed number of steps). 

Therefore, we can let a image smaple(from real data distribution)
$$ \mathbf{x}_0 \sim q(\mathbf{x}) $$

Usually we will also define steps
$$T$$, where step size is controlled by 
$$ \{\beta_t \in (0, 1)\}_{t=1}^T $$

With steps 
$$T$$
, we can produce a sequence of noisy samples 
$$ x_1, x_2, ..., x_T $$


