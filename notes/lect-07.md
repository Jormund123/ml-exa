# Lecture 07 — Gaussian Process Regression

**Source:** `slides/lect-07.txt` (71 slides)
**Exam mapping:** Directly covers **Exercise 7** (15 pts). Also supports **Exercise 5** (kernel trick, 10 pts).
**Priority:** CRITICAL — Exercise 7 is worth 15 pts and is one of the two highest-value exercises on the exam.

---

## Overview & Exam Mapping

| Slide Content | Maps to Exam Exercise | Priority |
|---|---|---|
| Recap of Bayesian regression (slides 4-15) | Reinforces Ex 3 (lect-05/06) | Already covered |
| Building GP from Bayesian regression, kernel trick (slides 16-42) | **Ex 5** (kernel trick), **Ex 7** (GP understanding) | HIGH |
| **GP log-likelihood and training (slides 43-55)** | **Ex 7b** (10 pts) | **CRITICAL** |
| **GP prediction formulas (slides 57-69)** | **Ex 7b** (10 pts) | **CRITICAL** |

---

## What is a Gaussian Process? (Conceptual Foundation)

**Intuition:** Instead of fitting a parameterized function $f(x \mid w)$ by finding optimal parameters $w$, a GP places a **distribution over functions** directly. We don't choose a specific function — we describe our beliefs about what the function looks like via a **kernel function** $k(x, x')$.

**Formally (slide 35):** A function $f$ can be thought of as an infinite-dimensional random vector, where $f(x)$ is the "$x$-th entry." If we assume this random vector is normally distributed:

$$f \sim \mathcal{N}\left(f_0, \, k(x, x')\right)$$

where $f_0(x) = 0$ is the mean function and $k(x, x')$ is a covariance (kernel) function.

For any finite set of inputs $\{x_1, \ldots, x_n\}$, we get a finite-dimensional projection:

$$\tilde{y} = \begin{bmatrix} f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix} \sim \mathcal{N}(0, K)$$

where $K_{ij} = k(x_i, x_j)$ is the **kernel matrix** (also called Gram matrix).

> **Key insight:** The model parameter $w$ is **gone**. Instead, a kernel $k(\cdot, \cdot)$ has appeared. Different kernels define different families of functions (slide 30). This is the kernel trick in action (→ Ex 5).

---

## How a GP is Built (From Bayesian Regression to GP)

This derivation (slides 20-30) shows why GPs are a natural extension of Bayesian regression. Understanding this chain helps answer "what is a GP?" questions.

### Step 1: Start with Bayesian regression

Model: $y_j = \phi_j^\top w + \epsilon_j$ with prior $w \sim \mathcal{N}(0, \sigma_0^2 I)$.

The prediction vector $\tilde{y} = \Phi^\top w$ is also Gaussian (since it's a linear transform of a Gaussian).

### Step 2: Compute mean and covariance of predictions

Since $E[w] = 0$:

$$E[\tilde{y}] = \Phi^\top E[w] = 0$$

Since $E[ww^\top] = \sigma_0^2 I$:

$$\text{cov}(\tilde{y}, \tilde{y}) = \Phi^\top E[ww^\top] \Phi = \sigma_0^2 \Phi^\top \Phi$$

So: $\tilde{y} \sim \mathcal{N}(0, \sigma_0^2 \Phi^\top \Phi)$.

### Step 3: Apply the kernel trick

The covariance matrix $\sigma_0^2 \Phi^\top \Phi$ is a Gram matrix with entries $(\Phi^\top \Phi)_{ij} = \phi_i^\top \phi_j$.

We **replace** this with a general kernel matrix $K$ where $K_{ij} = k(x_i, x_j)$:

$$\tilde{y} \sim \mathcal{N}(0, K)$$

> **This is the kernel trick (→ Ex 5):** Wherever we see inner products $\phi_i^\top \phi_j$, we replace them with a kernel evaluation $k(x_i, x_j)$. This allows us to work with potentially infinite-dimensional feature spaces without ever computing $\phi$ explicitly.

---

## The GP Model With Noise (→ Ex 7b)

The predictions $\tilde{y}$ are noiseless. Real observations have noise: $y_j = \tilde{y}_j + \epsilon_j$ where $\epsilon_j \sim \mathcal{N}(0, \sigma^2)$.

The marginal distribution of the noisy observations (slide 48):

$$p(y) = \int \underbrace{\mathcal{N}(y \mid \tilde{y}, \sigma^2 I)}_{p(y \mid \tilde{y})} \cdot \underbrace{\mathcal{N}(\tilde{y} \mid 0, K)}_{p(\tilde{y})} \, d\tilde{y}$$

This is a product of two Gaussians integrated — the result is Gaussian:

$$\boxed{y \sim \mathcal{N}(0, K + \sigma^2 I)}$$

We define the **covariance matrix** $C = K + \sigma^2 I$, which combines the kernel covariance $K$ with the noise variance $\sigma^2 I$.

> **Why $\sigma^2 I$ is added:** The kernel matrix $K$ captures the function's structure (how outputs covary based on input similarity). The $\sigma^2 I$ term models observation noise — each measurement has independent Gaussian noise.

---

## Training a GP: The Log-Likelihood (→ Ex 7b, CRITICAL)

**This is one of the most important formulas for Exercise 7b.** The exam asks: "How can you get the optimal parameters for $C(\theta)$? Write down explicit formulas."

### The Kernel and its Hyperparameters

A common kernel for GP regression (slide 50):

$$k(x_i, x_j) = \theta_0 \exp\left(-\frac{(x_i - x_j)^2}{2\theta_1^2}\right) + \theta_2 \, x_i x_j$$

This combines a **Gaussian (RBF) kernel** (controls smoothness) with a **linear kernel** (allows linear trends).

With noise parameter $\theta_3 = \sigma$, the full covariance matrix is:

$$[C(\theta)]_{ij} = \theta_0 \exp\left(-\frac{(x_i - x_j)^2}{2\theta_1^2}\right) + \theta_2 \, x_i x_j + \theta_3^2 \, \delta_{ij}$$

where $\delta_{ij}$ is the Kronecker delta (1 if $i=j$, 0 otherwise) and $\theta = (\theta_0, \theta_1, \theta_2, \theta_3)^\top$ are the **hyperparameters**.

### The GP Log-Likelihood

Given $y \sim \mathcal{N}(0, C)$, the log-likelihood of the hyperparameters is:

$$\boxed{\mathcal{L}(\theta) = \log p(y \mid \theta) = -\frac{1}{2}\log\det(C) - \frac{1}{2}y^\top C^{-1} y + \text{const}}$$

> **This is the formula the exam asks you to write for Ex 7b. Memorize it.** The three terms have clear interpretations:
> - $-\frac{1}{2}\log\det(C)$: **complexity penalty** — penalizes overly flexible models (large $\det C$)
> - $-\frac{1}{2}y^\top C^{-1}y$: **data fit** — measures how well the model explains the data
> - const: $-\frac{n}{2}\log(2\pi)$ — doesn't depend on $\theta$, can be ignored for optimization

### The MLE for Hyperparameters

$$\theta_{\text{ML}} = \arg\max_\theta \mathcal{L}(\theta) = \arg\max_\theta \left[-\frac{1}{2}\log\det(C) - \frac{1}{2}y^\top C^{-1}y\right]$$

### How to Solve: Gradient Ascent (→ Ex 7b, CRITICAL)

**There is no closed-form solution** for $\theta_{\text{ML}}$ (slide 52). We must use iterative optimization.

The gradient of the log-likelihood w.r.t. each hyperparameter $\theta_l$ is (slide 53):

$$\frac{\partial \mathcal{L}}{\partial \theta_l} = -\frac{1}{2}\text{tr}\left(C^{-1}\frac{\partial C}{\partial \theta_l}\right) + \frac{1}{2}y^\top C^{-1}\frac{\partial C}{\partial \theta_l}C^{-1}y$$

We optimize using **gradient ascent**:

$$\theta_{t+1} \leftarrow \theta_t + \eta_t \cdot \nabla_{\theta_t}\mathcal{L}$$

### The Non-Convexity Problem (→ Ex 7b, 3-5 pts)

**The log-likelihood $\mathcal{L}(\theta)$ is NOT convex** in the hyperparameters $\theta$.

This means gradient ascent can get stuck in **local optima**. The quality of the result depends critically on the initial guess $\theta_0$.

> **The exam answer for "How would you solve this non-convex optimization problem?":**
>
> **Gradient ascent with multiple random restarts:**
> 1. Randomly initialize hyperparameters $\theta_0$ multiple times
> 2. Run gradient ascent from each initialization
> 3. Keep the solution with the highest log-likelihood
>
> This helps escape local optima by exploring different regions of the parameter space.

> **This answer is worth 3-5 points on its own.** Both 1A and 1B ask this exact question. The keywords are: "gradient ascent" (or gradient descent on the negative), "multiple random restarts," "non-convex," "local optima."

---

## GP Prediction (→ Ex 7b)

**This is the second key formula set for Exercise 7b.**

### Setup (slides 58-59)

- Training inputs: $x = (x_1, \ldots, x_n)^\top$ with outputs $y = (y_1, \ldots, y_n)^\top$
- Test inputs: $x_* = (x_1^*, \ldots, x_N^*)^\top$ — we want to predict $y_*$

Four kernel matrices:

| Matrix | Size | Definition | What it captures |
|--------|------|-----------|------------------|
| $K_{xx}$ | $n \times n$ | $(K_{xx})_{ij} = k(x_i, x_j)$ | Similarity between training inputs |
| $K_{x*}$ | $n \times N$ | $(K_{x*})_{iq} = k(x_i, x_q^*)$ | Similarity between training and test inputs |
| $K_{*x}$ | $N \times n$ | $(K_{*x})_{pj} = k(x_p^*, x_j)$ | $= K_{x*}^\top$ |
| $K_{**}$ | $N \times N$ | $(K_{**})_{pq} = k(x_p^*, x_q^*)$ | Similarity between test inputs |

### The Joint Distribution (slide 60)

Training and test outputs follow a joint Gaussian:

$$\begin{bmatrix} y \\ y_* \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} K_{xx} + \sigma^2 I & K_{x*} \\ K_{*x} & K_{**} \end{bmatrix}\right)$$

### The Predictive Distribution (slides 63-64)

Conditioning on the observed $y$, the test outputs follow:

$$\boxed{y_* \sim \mathcal{N}(\mu_*, \Sigma_*)}$$

where:

$$\boxed{\mu_* = K_{*x}(K_{xx} + \sigma^2 I)^{-1}y}$$

$$\boxed{\Sigma_* = K_{**} - K_{*x}(K_{xx} + \sigma^2 I)^{-1}K_{x*}}$$

> **These are the formulas the exam asks you to write. Memorize both.**

**What each part means:**
- $\mu_*$ is the **predicted mean** — the GP's best guess for the test outputs. It's a weighted combination of the training outputs $y$, where the weights come from the kernel similarities between test and training inputs.
- $\Sigma_*$ is the **predictive covariance** — captures uncertainty. It starts from the prior covariance $K_{**}$ and **subtracts** a term that accounts for information gained from training data.

**Key properties of GP predictions:**
- **Near training data:** $K_{*x}(K_{xx} + \sigma^2 I)^{-1}K_{x*}$ is large → $\Sigma_*$ is small → **low uncertainty**
- **Far from training data:** $K_{*x}$ entries are small → $\Sigma_*$ is close to $K_{**}$ → **high uncertainty** (reverts to prior)
- **Predictions interpolate training data** (approximately, up to noise $\sigma^2$)

### Single Test Point (slide 68)

For a single test input $x_*$, define the kernel vector:

$$k_* = \begin{bmatrix} k(x_1, x_*) \\ k(x_2, x_*) \\ \vdots \\ k(x_n, x_*) \end{bmatrix}$$

Then:

$$E[y_* \mid x_*, x, y] = k_*^\top(K_{xx} + \sigma^2 I)^{-1}y = \sum_{j=1}^{n} k(x_j, x_*) \cdot \alpha_j$$

where $\alpha = (K_{xx} + \sigma^2 I)^{-1}y \in \mathbb{R}^n$.

> **Connection (slide 69):** This is an **RBF regression model** — the GP prediction is a weighted sum of kernel evaluations centered at training points. This connects GP regression back to kernel regression from lect-02 and lect-12.

---

## Exercise 7a: Handling Non-Zero Mean (5 pts)

**The exam question:** "We often model with $\mathcal{N}(0, C(\theta))$, but you cannot always assume a zero-mean. How do you work with arbitrary $y$?"

The GP model assumes $y \sim \mathcal{N}(0, C)$, which implies zero-mean outputs. Real data typically has non-zero mean.

**The solution: Center the data.**

1. Compute the sample mean: $\bar{y} = \frac{1}{n}\sum_{j=1}^{n} y_j$
2. Center the targets: $y' = y - \bar{y}\mathbf{1}$ (subtract mean from all targets)
3. Fit the GP to centered data: $y' \sim \mathcal{N}(0, C(\theta))$
4. For predictions, add the mean back: $\hat{y}_* = \mu_* + \bar{y}$

where $\mu_* = K_{*x}(K_{xx} + \sigma^2 I)^{-1}y'$ is the GP prediction on centered data.

> **What to write on the exam for Ex 7a (5 pts):**
>
> "Subtract the sample mean $\bar{y} = \frac{1}{n}\sum_j y_j$ from all training targets to get centered data $y' = y - \bar{y}\mathbf{1}$. Fit the GP to $y'$, which now has zero mean. When predicting for new inputs, add $\bar{y}$ back to the GP prediction: $\hat{y}_* = \mu_* + \bar{y}$."

> **Connection to lect-08 (normalization):** This centering is the same idea as data normalization/standardization. The GP's zero-mean assumption is not a limitation — it just requires pre-processing.

---

## Exercise 7b: Complete Answer Template (10 pts)

**The exam asks for all of the following. Write them in this order for maximum partial credit.**

### Part 1: The kernel and covariance matrix (2-3 pts)

"A common kernel for GP regression is:

$$k(x_i, x_j) = \theta_0 \exp\left(-\frac{(x_i - x_j)^2}{2\theta_1^2}\right) + \theta_2 \, x_i x_j$$

The covariance matrix is $C(\theta) = K + \theta_3^2 I$, where $K_{ij} = k(x_i, x_j)$ and $\theta_3^2$ is the noise variance."

### Part 2: The log-likelihood (3-4 pts)

"The log-likelihood of the hyperparameters $\theta$ is:

$$\mathcal{L}(\theta) = -\frac{1}{2}\log\det(C) - \frac{1}{2}y^\top C^{-1}y + \text{const}$$

We find optimal hyperparameters via: $\theta_{\text{ML}} = \arg\max_\theta \mathcal{L}(\theta)$."

### Part 3: How to solve — non-convex optimization (3-4 pts)

"This optimization problem is **not convex** — the log-likelihood has multiple local optima. We solve it using **gradient ascent with multiple random restarts**:

1. Choose multiple random initializations $\theta_0$
2. For each, run gradient ascent: $\theta_{t+1} \leftarrow \theta_t + \eta \nabla_\theta \mathcal{L}$
3. Keep the $\theta$ that achieves the highest $\mathcal{L}(\theta)$

The gradient is: $\frac{\partial \mathcal{L}}{\partial \theta_l} = -\frac{1}{2}\text{tr}(C^{-1}\frac{\partial C}{\partial \theta_l}) + \frac{1}{2}y^\top C^{-1}\frac{\partial C}{\partial \theta_l}C^{-1}y$"

> **Partial credit strategy:** Even if you can't write the gradient, writing the log-likelihood and saying "gradient ascent with multiple random restarts because the problem is non-convex" earns you 6-7 of the 10 points. **Never leave this blank.**

---

## Computational Cost of GP (→ Ex 7, "reason about" questions)

From Exam 1C and general GP knowledge:

- GP requires inverting the $n \times n$ matrix $(K_{xx} + \sigma^2 I)$ → **cost is $O(n^3)$**
- This makes GPs expensive for large datasets
- The kernel matrix must be **positive (semi-)definite** — this is guaranteed by using Mercer kernels

---

## Connections to Other Exercises

| This Lecture Concept | Feeds Into |
|---------------------|------------|
| Kernel trick: $\sigma_0^2\Phi^\top\Phi \to K$ | **Ex 5** (what is the kernel trick?) |
| GP log-likelihood as optimization objective | **Ex 7b** (write the objective, apply meta-pattern) |
| Non-convex → gradient ascent + random restarts | **Ex 7b** (how to solve?), also applies to EM, neural nets |
| GP prediction = weighted sum of kernels | Connection to kernel regression (lect-12) |
| $C = K + \sigma^2 I$ (noise added to kernel) | **Ex 7** (what is $C(\theta)$?) |
| Posterior predictive with uncertainty bands | **Ex 3c** (posterior predictive concept) |

---

## What to Skip in This Lecture

- **Detailed recap of Bayesian regression** (slides 4-15) — already covered in lect-05/06 notes
- **The "infinite-dimensional vector" philosophical discussion** (slides 34-36) — nice for intuition, zero exam points
- **The illustration slides** (slides 39-42, 54-55, 65-66) — visual examples, not tested
- **The proof that GP prediction = RBF regression** (slides 68-69) — know the connection exists, don't memorize

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 07:

| # | Formula | Maps to | Memorize? |
|---|---------|---------|-----------|
| 1 | $y \sim \mathcal{N}(0, K + \sigma^2 I)$ | **Ex 7** (GP model) | **YES** |
| 2 | $\mathcal{L}(\theta) = -\frac{1}{2}\log\det(C) - \frac{1}{2}y^\top C^{-1}y + \text{const}$ | **Ex 7b** (log-likelihood) | **YES — write this verbatim** |
| 3 | $\mu_* = K_{*x}(K_{xx} + \sigma^2 I)^{-1}y$ | **Ex 7b** (GP mean prediction) | **YES — write this verbatim** |
| 4 | $\Sigma_* = K_{**} - K_{*x}(K_{xx} + \sigma^2 I)^{-1}K_{x*}$ | **Ex 7b** (GP covariance prediction) | **YES — write this verbatim** |
| 5 | $k(x_i, x_j) = \theta_0 e^{-(x_i - x_j)^2 / (2\theta_1^2)} + \theta_2 x_i x_j$ | **Ex 7b** (kernel for $C(\theta)$) | **YES** |
| 6 | Center $y$ by subtracting $\bar{y}$ for non-zero mean | **Ex 7a** (5 pts) | **YES** |
| 7 | "Gradient ascent with multiple random restarts" | **Ex 7b** (how to solve non-convex) | **YES — this phrase alone is worth 3-5 pts** |
| 8 | GP costs $O(n^3)$ from inverting $n \times n$ matrix | **Ex 7** (Exam 1C style) | YES |

### Verdict:

**Drill formulas 1-7 until you can write them blind.** Exercise 7 is worth 15 points — the second-highest-value exercise on the exam. Formulas 2, 3, and 4 are the core. The phrase "gradient ascent with multiple random restarts because the problem is non-convex" is a guaranteed 3-5 points whenever a non-convex optimization question appears.
