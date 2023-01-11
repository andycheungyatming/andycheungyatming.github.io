---
layout:       post
title:        "Code Review: Denoising Diffusion Probabilistic Models (DDPM)"
author:       "Allan"
header-style: text
catalog:      true
mathjax:      true
publish:      false
tags:
    - AI
    - Code
    - Review
    - Diffusion
    - Generation
---

# Introduction

DDPM maths background and introduction can refer to [Review: Denoising Diffusion Probabilistic Models (DDPM)](2022-12-21-Review-DDPM.md)

In short, we have a the following training algorithm to be implemented 

![](https://huggingface.co/blog/assets/78_annotated-diffusion/training.png)

# Model Backbone

A neural network is applied to implement the reverse process. Technically, the network takes in and outputs tensors of the same shape. Authors has chosen U-Net [(Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) as backbone.

Also, U-Net also introduced residual connections between the encoder and decoder, making a better gradient flow, inspired by ResNet [(He et al., 2015)](https://arxiv.org/abs/1512.03385).

a U-Net model first downsamples the input (i.e. makes the input smaller in terms of spatial resolution), after which upsampling is performed.

![](https://huggingface.co/blog/assets/78_annotated-diffusion/unet_architecture.jpg)

## Positioning Encoding 
Inspired by the Transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), as the parameters of the neural network are shared across time (noise level), the authors employ sinusoidal position embeddings to encode $t$. This makes the neural network "know" at which particular time step (noise level) it is operating, for every image in a batch.

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

## Resnet block

Next, we define the core building block of the U-Net model. The DDPM authors employed a Wide ResNet block [(Zagoruyko et al., 2016)](https://arxiv.org/abs/1605.07146), but Phil Wang has replaced the standard convolutional layer by a "weight standardized" version, which works better in combination with group normalization (see [(Kolesnikov et al., 2019)](https://arxiv.org/abs/1912.11370) for details).

```python
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
```

## Attention
Next, we define the attention module, which the DDPM authors added in between the convolutional blocks. Attention is the building block of the famous Transformer architecture [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), which has shown great success in various domains of AI, from NLP and vision to protein folding. Phil Wang employs 2 variants of attention: one is regular multi-head self-attention (as used in the Transformer), the other one is a linear attention variant [(Shen et al., 2018)](https://arxiv.org/abs/1812.01243), whose time- and memory requirements scale linear in the sequence length, as opposed to quadratic for regular attention.

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
```

## Group Normalization 
The DDPM authors interleave the convolutional/attention layers of the U-Net with group normalization.

We define a PreNorm class, which will be used to apply groupnorm before the attention layer, as we'll see further. 

Remark: there's been a debate about whether to apply normalization before or after attention in [Transformers without Tears](https://tnq177.github.io/data/transformers_without_tears.pdf) (Nguyen & Salazar, 2019).

```python
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
```

## Conditional U-Net
Now that we've defined all building blocks (position embeddings, ResNet blocks, attention and group normalization), it's time to define the entire neural network.

The network takes a batch of noisy images of shape `(batch_size, num_channels, height, width)` and a batch of noise levels of shape `(batch_size, 1)` as input, and returns a tensor of shape `(batch_size, num_channels, height, width)`

Recall the U-Net architecture.
![](https://huggingface.co/blog/assets/78_annotated-diffusion/unet_architecture.jpg)

