---
layout:       post
title:        "Review: Denoising Diffusion Probabilistic Models (DDPM)"
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
# What are Diffusion Models?

Denoising Diffusion Probabilistic Models (DDPM) is introduced in [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239). The maths background is discussed [(Sohl-Dickstein et al., 2015)](https://arxiv.org/abs/1503.03585). The essential idea is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process which is fixed.

# DDPM General Explanation

![figure1](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

> The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise.

we can let a image smaple(from real data distribution)

$$
\mathbf{x}_0 \sim q(\mathbf{x})
$$

Usually we will also define steps

$$
T
$$

, where step size is controlled by

$$
\{\beta_t \in (0, 1)\}_{t=1}^T
$$

With steps

$$
T
$$

, we can produce a sequence of noisy samples

$$
x_1, x_2, ..., x_T = z
$$

However, it is difficult to rebuild the image from

$$
x_t
$$

 to

$$
x_0
$$

 directly, we need the model to learn the rebuilt process piece by piece.
i.e.

$$
x_T \rarr x_{T-1} = u(x_T) \rarr x_{T-2} = u(x_{T-1}) \rarr ... \rarr x_1 = u(x_2) \rarr x_0 = u(x_1)
$$

## Destruction (Forward Process)

I'd like to use destruction instead of forward process. Basically we want to make a image (with pattern) to a pure gaussian noise by putting more gaussian noise recursively (with a fixed number of steps).

### Merging of Gaussian Noise

Two Gaussian ,e.g.

$$
\boldsymbol{N}(0,\sigma^2_1 \boldsymbol{I}) \And  \boldsymbol{N}(0,\sigma^2_2 \boldsymbol{I})
$$

with different variance can be merged to

$$
\boldsymbol{N}(0,(\sigma^2_1+\sigma^2_2) \boldsymbol{I})
$$

Each step can be defined as following formulas.

### Details on destruction process

we define each step of

$$
\boldsymbol{x}_t
$$

as

$$
\boldsymbol{x}_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t , \text{where } \epsilon\sim \boldsymbol{N}(0, \boldsymbol{I})
$$

where

$$
\alpha_t + \beta_t = 1 \text{ and } \beta \approx 0
$$

and let

$$
\bar{\alpha}_t = \prod^t_{i=1}\alpha_i
$$

, we have

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon_{t-1}}\\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\boldsymbol{\epsilon_{t-2}} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon_{t-1}}\\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{(\sqrt{\alpha_t(1 - \alpha_{t-1})}\boldsymbol{\epsilon_{t-2}})^2 + (\sqrt{1-\alpha_t}\boldsymbol{\epsilon_{t-1}})^2}\\
&=\sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{\cancel{\alpha_t} - \alpha_t\alpha_{t-1}+1 - \cancel{\alpha_t} }\bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).}\\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2}  \\
&= \dots \\
&= \sqrt{(a_t\dots a_1)} \mathbf{x}_0 + \sqrt{1 - (a_t\dots a_1)}\boldsymbol{\bar{\epsilon}}\\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\bar{\epsilon}} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \\
\text{and }q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
\end{aligned}
$$

where

$$
\boldsymbol{\bar{\epsilon}}
$$

is a sum of i.i.d gaussian noise

Therefore, we can observe by more steps iterated, the more image will be converted to pure noise.

### Merging of two Gaussian

Two Gaussian ,e.g.

$$
\mathbb{N}(0,\sigma^2_1 \boldsymbol{I}) \And  \mathbb{N}(0,\sigma^2_2 \boldsymbol{I})
$$

with different variance can be merged to

$$
\mathbb{N}(0,(\sigma^2_1+\sigma^2_2) \boldsymbol{I})
$$

#### Proof

Recall the theorem of variance.

$$
\mathbb{V}(X+Y) = \mathbb{V}(X) + \mathbb{V}(Y) + 2\mathbb{Cov}(XY)
$$

More generally, for random variables

$$
X_1, X_2, \dots, X_N
$$

we have

$$
\mathbb{V}\left(\sum_{i=1}^N a_i X_i\right)=\sum_{i=1}^N a_i^2 \mathbb{V}\left(X_i\right)+2 \sum_{i=1}^{N-1} \sum_{j=i+1}^N a_i a_j \operatorname{Cov}\left(X_i, X_j\right)
$$

As

$$
X_i \sim \mathbb{N}(\mu_i, \Sigma_i), i=1 \cdots N,
$$

, we have claim they are i.i.d. Hence the covariances are zero. i.e.

$$
\textbf{Cov}(X_i,X_j)=0
$$

A newly aggregated these Gaussian distributions is defined as a weighted sum:

$$
X=\sum_{i=1}^{N} a_i X_i=\sum_{i=1}^{N} \frac{n_i}{\sum_{l=1}^{N}n_l} X_i, \\ 
\text{where} \sum_{i=1}^N a_i = 1
$$

Therefore, we can find the merged variance as:

$$
\begin{aligned}
\text{Consider } \mathbb{X}_i &\sim \mathbb{N}(\mu_i, \Sigma_i) \\
\mu &= \sum_{i=1}^{N} a_i \mu_i \\ 
\Sigma &=\mathbb{V}(X)= \mathbb{V}(\sum_{i=1}^{N} a_i X_i)\\
&=\sum_{i=1}^{N}a_i^2\mathbb{V}(X_i) + 2 \sum_{i=1}^{N-1}\sum_{j=i+1}^{N} a_i a_j \textbf{Cov}(X_i,X_j) \\
& = \sum_{i=1}^{N}a_i^2 \Sigma_i + 0 \\
& = \sum_{i=1}^{N}a_i^2 \Sigma_i \\

\end{aligned}
$$

### Schedule

The formula

$$
\bar{\alpha}_t = \prod^t_{i=1}\alpha_i
$$

is following a schedule. The schedle is responsilbe to how the way is to destruct an image to pure noise.

#### Linear Schedule

The DDPM adopt linear schedule as follows:

```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
```

![linear schedule](https://i.imgur.com/Y5HARtf.png)

#### Cosine Schedule

![cosine schedule](https://i.imgur.com/dj9bcqr.png)
Later cosine schedule is proponsed. It replace the linear schedule as:

- In linear schedule, the last couple of timesteps already seems like complete noise
- and might be redundent.
- Therefore, Information is destroyed too fast.

Cosine schedule can solve the problem mentioned above.

### Connection with frequency and destruction

![Illustration of forward process](https://i.imgur.com/OdL8Agc.png)

- At small t, most of the low frequency contents are not perturbed by the noise, but high frequency content are being perturbed.
- At bigger t, low frequency contents are also perturbed.
- At the end of forward process, we get rid of the both low and high frequency contents of image.
  ![Freqency connection](https://i.imgur.com/l3f0Wo3.png)

## Building (Reverse Process)

![圖 3](https://s2.loli.net/2022/12/12/ZQoaAqjb2f64erd.png)

We know how to process the forward process. However we have no idea to recover an image from noise as we dont know the formula, or equation for it. Luckily we can use deep neural network to approximate one due to [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

However, it is mentioned that it is difficult to recover/ generate directly from

$$
x_t \rarr x_0
$$

. Therefore, the intuitive idea is to find

$$
q(x_{t-1}|x_t)
$$

repeatedly and remove the noise (denoise) the image piece by piece. i.e.

$$
\begin{aligned}
\boldsymbol{x}_{t-1} &= \frac{1}{\sqrt{\alpha_t}}(\boldsymbol{x}_t - \sqrt{\beta_t}\epsilon_t) \\
\therefore 
\boldsymbol{\mu}(\boldsymbol{x}_t) &= \frac{1}{\sqrt{\alpha_t}}\left(\boldsymbol{x}_t   - \sqrt{\beta_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)
\end{aligned}
$$

where

$$
\theta
$$

is trainable parameter

Therefore, we can focus on and formulate a loss function by minimising predicted output

$$
u(x_t)
$$

 and groundtruth

$$
x_{t-1}
$$

as

$$
\begin{aligned}
\mathbb{L} &= \left\Vert\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2 \\
&= \left\Vert\frac{1}{\sqrt{\alpha_t}}(\boldsymbol{x}_t - \sqrt{\beta_t}\epsilon_t) - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2 \\
&= \frac{\beta_t}{\alpha_t}\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right\Vert^2 \\
&= \frac{\beta_t}{\alpha_t}\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{\bar{\beta}}\boldsymbol{\bar{\epsilon}}, t)\right\Vert^2
\end{aligned}
$$

which is similar to loss function in DDPM [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

$$
\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t\left(1-\bar{\alpha}_t\right)}\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t\right)\right\|^2\right]
$$

Therefore, we can train an network by minisming the captioned loss function, and generate a random image starting by

$$
x_T \sim \boldsymbol{N}(0, \boldsymbol{I})
$$

to

$$
x_0
$$

via

$$
\boldsymbol{\mu}(\boldsymbol{x}_t) = x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(\boldsymbol{x}_t   - \sqrt{\beta_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\big)
$$

which is simply an expression of

$$
x_{t-1} \approx x_t - noise, \text{where } noise \sim \boldsymbol{N}(0,I)
$$

in each step.

# DDPM Explanation by Bayes' Perspective

### ~~Product of Gaussian~~

Let

$$
f(x) \text{ and } g(x)
$$

be Gaussian PDFs, where

$$
f(x)=\frac{1}{\sqrt{2 \pi} \sigma_f} e^{-\frac{\left(x-\mu_f\right)^2}{2 \sigma_f^2}} \text { and } g(x)=\frac{1}{\sqrt{2 \pi} \sigma_g} e^{-\frac{\left(x-\mu_g\right)^2}{2 \sigma_g^2}}
$$

### Completing the Square

We have a quadratic form as follows:
![圖 4](https://s2.loli.net/2022/12/12/hD3uCm7ZF9QqxXA.png)

$$
\begin{aligned}
    ax^2 + bx + c &= 0 \\
    a(x+d)^2 +e &= 0 \\
    \therefore d &= \frac{b}{2a} \\
    \text{And  } e &= c - \frac{b^2}{4a} = c - d^2
\end{aligned}
$$

## Recall of Bayes' Theorem

[Bayes&#39; theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) is defined as follows:

$$
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \frac{p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})p(\boldsymbol{x}_{t-1})}{p(\boldsymbol{x}_t)}\end{equation}
$$

However, we do not know the expression formulas of

$$
p(\boldsymbol{x}_{t-1}),p(\boldsymbol{x}_t)
$$

. However, we can add one more condition

$$
\boldsymbol{x}_0
$$

s.t.

$$
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{p(\boldsymbol{x}_t|\boldsymbol{x}_0)}
\end{equation}
$$

where

$$
p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}),p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0),p(\boldsymbol{x}_t|\boldsymbol{x}_0)
$$

is well known.

Therefore, we can get the following expression:

$$
\begin{aligned}
    p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) &= \boldsymbol{N}(\boldsymbol{x}_t; \sqrt{\alpha_t}\boldsymbol{x}_{t-1};\beta_t\boldsymbol{I}) \\
    p(\boldsymbol{x}_{t-1} \vert \boldsymbol{x}_0) &= \boldsymbol{N}(\boldsymbol{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} 
    \boldsymbol{x}_0, \bar{\beta}_{t-1}\boldsymbol{I}) \\
    p(\boldsymbol{x}_t \vert \boldsymbol{x}_0) &= \boldsymbol{N}(\boldsymbol{x}_t; \sqrt{\bar{\alpha}_t} 
    \boldsymbol{x}_0, \bar{\beta}_t\boldsymbol{I})

\end{aligned}
$$

We put all expression above to compute:

$$
\begin{aligned}
p(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
&= p(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ p(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ p(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)} \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)} \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)} \\
&= exp\Big(-\frac{1}{2}\big(\color{red}{a}\color{black}{(x+}\color{blue}{d})^2 \color{black}{+ e} \big)\Big) \\
&= \boldsymbol{N}( d , \frac{1}{a}\boldsymbol{I}) 
\end{aligned}
$$

Recall that

$$
f(x)=\frac{1}{\sqrt{2 \pi} \sigma_f} exp\Big({-\frac{1}{2}\frac{\left(x-\color{black}{\mu_f}\right)^2}{\color{black}{\sigma_f^2}}}\Big) = \boldsymbol{N}(\mu_f, \sigma^2_f)
$$

We can focus on getting the coefficient of

$$
\boldsymbol{x}_{t-1}^2
$$

as this term is quadratic, this implies the final format also follow gaussian distribution. We get the quadratic form by [completing the square](https://www.mathsisfun.com/algebra/completing-square.html):

$$
\begin{aligned}
a = \frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta}_{t-1}} = \frac{\alpha_t\bar{\beta}_{t-1} + \beta_t}{\bar{\beta}_{t-1} \beta_t} &= \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\bar{\beta}_{t-1} \beta_t} = \frac{1-\bar{\alpha}_t}{\bar{\beta}_{t-1} \beta_t} = \frac{\bar{\beta}_t}{\bar{\beta}_{t-1} \beta_t} \\
\therefore p(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) &= \boldsymbol{N}( d , \frac{\bar{\beta}_{t-1} \beta_t}{\bar{\beta}_t}\boldsymbol{I}
)

\end{aligned}
$$

Therefore, 

$$
\begin{aligned}
d = \frac{b}{2a}
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
&= \frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t}\boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{\bar{\beta}_t}\boldsymbol{x}_0 \\
\end{aligned}
$$

We can get the final distribution expression s.t. 

$$
\begin{equation} 
\therefore p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}\left(\boldsymbol{x}_{t-1};\frac{\sqrt{\alpha_t}\bar{\beta}_{t-1}}{\bar{\beta}_t}\boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{\bar{\beta}_t}\boldsymbol{x}_0,\frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t} \boldsymbol{I}\right)
\end{equation}
$$

