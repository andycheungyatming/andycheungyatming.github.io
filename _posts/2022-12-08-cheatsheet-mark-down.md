---
layout: post
title: "Cheatsheet: Markdown"
author: "Allan"
header-style: text
catalog: true
mathjax: true
tags: 
    - cheatsheet
    - markdown
---
This is a very simple markdown cheatsheet for me to recap all command and syntax to write a markdown file.

# Title

```markdown
# Title
```

## Subtitle

```markdown
## Subtitle
```

### Smaller subtitle

```markdown
### Smaller subtitle
```

# Table

| Number | Next number | Previous number |
| :----- | :---------- | :-------------- |
| Five   | Six         | Four            |
| Ten    | Eleven      | Nine            |
| Seven  | Eight       | Six             |
| Two    | Three       | One             |

```markdown
| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |
```

# Photo

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

```markdown
![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)
```

# Code

```js
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

```markdown
~~~javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~
```

# Link

[Github](http://www.github.com/)

```md
[Github](http://www.github.com/)
```

# Quotes

> Imagination is more important than knowledge.
>
> Albert Einstein

```md
> Imagination is more important than knowledge.
>
> Albert Einstein
```

# YAML header

used to add as markdown top to serve as header

```yaml
---
title: "Page Title"
subtitle: "Page sub-title"
author: "Author name"
description: "This is a test"
institute: "MU"
date: "20/02/2020"
abstract: "YAML"
keywords: 
  - key1
  - key2
tags:
  - tag1
  - tag2
---
```

# Font

## Emphasis

*Italic* and **Bold**

```md
*Italic* and **Bold**
```

~~Scratched Text~~

```md
~~Scratched Text~~
```

## Font sizes

$$
\Huge Hello!
$$

$$
\huge Hello!
$$

$$
\LARGE Hello!
$$

$$
\Large Hello!
$$

$$
\large Hello!
$$

$$
\normalsize Hello!
$$

$$
\small Hello!
$$

$$
\scriptsize Hello!
$$

$$
\tiny Hello!
$$

```md
$$\Huge Hello!$$
$$\huge Hello!$$
$$\LARGE Hello!$$
$$\Large Hello!$$
$$\large Hello!$$
$$\normalsize Hello!$$
$$\small Hello!$$
$$\scriptsize Hello!$$
$$\tiny Hello!$$
```

# Latex

## Inline equation

$z = x\times y$

```
$ z = x\times y $
```

## Display equation

$$
z = x \times y
$$

```
$$
z = x \times y
$$
```

## Operators

- $$
  x + y
  $$
- $$
  x - y
  $$
- $$
  x \times y
  $$
- $$
  x \div y
  $$
- $$
  \dfrac{x}{y}
  $$
- $$
  \sqrt{x}
  $$

```latex
- $$x + y$$
- $$x - y$$
- $$x \times y$$ 
- $$x \div y$$
- $$\dfrac{x}{y}$$
- $$\sqrt{x}$$
```

## Symbols

- $$
  \pi \approx 3.14159
  $$
- $$
  \pm \, 0.2
  $$
- $$
  \dfrac{0}{1} \neq \infty
  $$
- $$
  0 < x < 1
  $$
- $$
  0 \leq x \leq 1
  $$
- $$
  x \geq 10
  $$
- $$
  \forall \, x \in (1,2)
  $$
- $$
  \exists \, x \notin [0,1]
  $$
- $$
  A \subset B
  $$
- $$
  A \subseteq B
  $$
- $$
  A \cup B
  $$
- $$
  A \cap B
  $$
- $$
  X \implies Y
  $$
- $$
  X \impliedby Y
  $$
- $$
  a \to b
  $$
- $$
  a \longrightarrow b
  $$
- $$
  a \Rightarrow b
  $$
- $$
  a \Longrightarrow b
  $$
- $$
  a \propto b
  $$

```latex
- $$\pi \approx 3.14159$$
- $$\pm \, 0.2$$
- $$\dfrac{0}{1} \neq \infty$$
- $0 < x < 1$$
- $0 \leq x \leq 1$$
- $x \geq 10$$
- $$\forall \, x \in (1,2)$$
- $$\exists \, x \notin [0,1]$$
- $$A \subset B$$
- $$A \subseteq B$$
- $$A \cup B$$
- $$A \cap B$$
- $$X \implies Y$$
- $$X \impliedby Y$$
- $$a \to b$$
- $$a \longrightarrow b$$
- $$a \Rightarrow b$$
- $$a \Longrightarrow b$$
- $$a \propto b$$
```

## Examples

$$
\mathbb{N} = \{ a \in \mathbb{Z} : a > 0 \}
$$

```latex
$$\mathbb{N} = \{ a \in \mathbb{Z} : a > 0 \}$$
```

$$
\forall \; x \in X \quad \exists \; y \leq \epsilon
$$

```latex
$$\forall \; x \in X \quad \exists \; y \leq \epsilon$$
```

$$
\color{blue}{X \sim Normal \; (\mu,\sigma^2)}
$$

```latex
$$\color{blue}{X \sim Normal \; (\mu,\sigma^2)}$$
```

$$
f(x) = x^2 - x^\frac{1}{\pi}
$$

```latex
$$f(x) = x^2 - x^\frac{1}{\pi}$$
```

$$
f(x) = \sqrt[3]{2x} + \sqrt{x-2}
$$

```latex
$$f(x) = \sqrt[3]{2x} + \sqrt{x-2}$$
```

$$
\mathrm{e} = \sum_{n=0}^{\infty} \dfrac{1}{n!}
$$

```latex
$$\mathrm{e} = \sum_{n=0}^{\infty} \dfrac{1}{n!}$$
```

$$
\lim_{x \to 0^+} \dfrac{1}{x} = \infty
$$

```
$$\lim_{x \to 0^+} \dfrac{1}{x} = \infty$$
```

$$
\int_a^b y \: \mathrm{d}x
$$

```
$$\int_a^b y \: \mathrm{d}x$$
```

$$
\max(S) = \max_{i:S_i \in S} S_i
$$

```
$$\max(S) = \max_{i:S_i \in S} S_i$$
```

$$
\dfrac{n!}{k!(n-k)!} = \binom{n}{k}
$$

```
$$\dfrac{n!}{k!(n-k)!} = \binom{n}{k}$$
```

## Functions

$$
f(x)=
\begin{cases}
1/d_{ij} & \quad \text{when $d_{ij} \leq 160$}\\ 
0 & \quad \text{otherwise}
\end{cases}
$$

```latex
$$ 
f(x)=
\begin{cases}
1/d_{ij} & \quad \text{when $d_{ij} \leq 160$}\\ 
0 & \quad \text{otherwise}
\end{cases}
$$
```

## Matrix

$$
M = 
\begin{bmatrix}
\frac{5}{6} & \frac{1}{6} & 0 \\[0.3em]
\frac{5}{6} & 0 & \frac{1}{6} \\[0.3em]
0 & \frac{5}{6} & \frac{1}{6}
\end{bmatrix}
$$

```latex
$$
M = 
\begin{bmatrix}
\frac{5}{6} & \frac{1}{6} & 0 \\[0.3em]
\frac{5}{6} & 0 & \frac{1}{6} \\[0.3em]
0 & \frac{5}{6} & \frac{1}{6}
\end{bmatrix}
$$
```

$$
M =
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
$$

```latex
$$ 
M =
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
$$
```

$$
A_{m,n} = 
\begin{pmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
\end{pmatrix}
$$

```latex
$$
A_{m,n} = 
\begin{pmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
\end{pmatrix}
$$
```

## Maths example with font size

$$
\small \text{Font size is small, eg.} \sum{x_i = 10}
$$

```latex
$$
\small \text{Font size is small, eg.} \sum{x_i = 10}
$$```
```
