---
layout:       post
title:        "Review: High-Resolution Image Synthesis with Latent Diffusion Models (LDM)"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
tags:
    - AI
    - Review
    - Diffusion
    - Generation
---
# Latent diffusion model (LDM)
Latent diffusion model(LDM) ([Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. 

It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent.

![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png)

# Latent representation from VAE
The perceptual compression process relies on an autoencoder model. An encoder $\mathcal{E}$ is used to compress the input image $x \in \mathbb{R}^{H\times W \times 3}$ to a smaller 2D latent vector $z = \mathcal{E}(x) \in \mathbb{R}^{h\times w \times c}$, where downsampling rate $f=H/h = W/w = 2^m, m \in \mathbb{N}$.

Then an decoder $\mathcal{D}$ reconstructs the images from the latent vector $\tilde{x} = \mathcal{D}(z)$.

Here the paper explored two types of regularization. 
- KL-reg: A small KL penalty towards a standard normal distribution over the learned latent, similar to [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- VQ-reg: Uses a vector quantization layer within the decoder, like [VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2) but the quantization layer is absorbed by the decoder.

# Diffusion in Latent Space
![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)
The diffusion and denoising processes happen on the latent vector $z$. The denoising model is a time-conditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image).

The design is equivalent to fuse representation of different modality into the model with cross-attention mechanism. 

Each type of conditioning information is paired with a domain-specific encoder $\tau_\theta$ to project the conditioning input $y$ to an intermediate representation that can be mapped into cross-attention component.

# Conditioned Generation
While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.

