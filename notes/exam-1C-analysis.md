# Exam 1C Analysis — Cross-Reference with Our Exam

**Source:** `past-questions/POML_2324_WS_1C.txt`
**IMPORTANT:** This is from a **DIFFERENT course** (University of Luebeck, Prof. Rueckert, "Probabilistic Machine Learning RO5601", Feb 2020). Different format: 4 questions, 100 points, 90 minutes, aids allowed.

**However:** Many topics overlap with our MA-INF 4111. Below I extract what's useful for OUR exam.

---

## 1C Exam Structure (different from ours)

| Q | Topic | Points |
|---|-------|--------|
| 1 | Probability theory (Gaussians, Bayes' theorem, Bayesian inference) | 25 |
| 2 | Linear probabilistic regression (derivatives, ridge regression, overfitting) | 25 |
| 3 | Nonlinear regression / Gaussian Processes (GP sketches, kernels, GP statements) | 25 |
| 4 | Probabilistic inference (marginalization, Gaussian identities) | 25 |

---

## What's Relevant for OUR Exam (MA-INF 4111)

### From Q1: Probability / Bayes (→ Our Exercise 3)

**Q1c (9 pts):** "State Bayes' Theorem and name all parts (Likelihood, posterior, prior) properly! Calculate the probability..."

This is exactly our Exercise 3a. The answer:

$$p(\theta|D) = \frac{p(D|\theta) \cdot p(\theta)}{p(D)}$$

- $p(\theta|D)$ = **posterior**
- $p(D|\theta)$ = **likelihood**
- $p(\theta)$ = **prior**
- $p(D)$ = **evidence** (marginal likelihood)

**Key learning from 1C:** The 1C exam asks you to apply Bayes to a CONCRETE scenario (Hepatitis B testing). Our exam is more abstract ("write down Bayes' theorem"). But knowing a concrete example strengthens your "reason about why Bayesian inference is hard" answer (Exercise 3d).

**Q1d (9 pts):** Bayesian inference with Beta-Binomial (casino/cheating). This tests conjugate priors and posterior computation. Maps to our Exercise 3b (conjugate priors) and 3d (why it's hard).

> **Takeaway for our exam:** If asked to "explain conjugate priors," having a concrete example (Beta prior + Binomial likelihood → Beta posterior) is strong.

---

### From Q2: Linear Regression / Ridge (→ Our Exercise 4 and connections)

**Q2b (5 pts):** "Derive the solution for $\mu$" from $K\mu = g$

This is our least squares derivation. Solution: $\mu = (K^\top K)^{-1}K^\top g$

**Q2c (4 pts):** "Write down ridge regression result"

$$\mu = (K^\top K + \lambda I)^{-1}K^\top g$$

**Why ridge regression is useful:** It adds $\lambda I$ to the diagonal, making the matrix always invertible (even if $K^\top K$ is singular) and preventing overfitting through regularization.

> **Takeaway for our exam:** Ridge regression = MAP with Gaussian prior. This connection is tested in Exercise 3 ("explain connection between regularization and MAP"). If the exam asks "why is regularization useful?" → prevents overfitting, equivalent to adding a Gaussian prior on parameters.

**Q2d (3 pts):** Illustrate overfitting, underfitting, optimal fit.

> **Takeaway for our exam:** If a "why" question about overfitting appears → underfitting = model too simple (high bias), overfitting = model too complex (high variance, fits noise). Regularization/kernel choice controls this trade-off.

---

### From Q3: Gaussian Processes (→ Our Exercise 7)

**Q3a (6 pts):** Identify mistakes in a GP sketch. Tests understanding of:
- GP mean passes through data points (with noise)
- Variance = 0 at observed points (without noise)
- Variance increases away from data

> **Takeaway for our exam:** GP predictions have uncertainty that GROWS as you move away from observed data. At data points, uncertainty is minimal.

**Q3b (10 pts):** True/False about GPs:
- "The covariance matrix K used for GPs has to be **positive definite**" → TRUE
- "$K_{ij} = \text{kernel}(x_i, x_j)$ defines a required covariance matrix if kernel is a valid kernel function" → TRUE
- "Gaussian processes are stochastic processes" → TRUE
- "GPs have the computational complexity of $O(N^3)$" (they said $N^4$ which is wrong) → the correct complexity is $O(N^3)$ due to matrix inversion

> **Takeaway for our exam:** The kernel matrix must be **positive definite** (this is what makes a kernel "valid"). GP computational cost is dominated by inverting the $N \times N$ covariance matrix → $O(N^3)$. These could be "reason about" points in Exercise 7.

**Q3c (4 pts):** Match kernel functions to GP sample plots. Tests understanding of:
- RBF kernel with small length-scale → wiggly functions
- RBF kernel with large length-scale → smooth functions
- Linear kernel → linear functions
- Exponential kernel → rough/non-smooth functions

> **Takeaway for our exam:** Know how kernel parameters affect GP behavior. For Exercise 7b, you need to explain the kernel $k(x_i, x_j) = \theta_1 \exp(-\|x_i - x_j\|^2 / (2\theta_2^2)) + \theta_3\delta_{ij}$. The length-scale $\theta_2$ controls smoothness. The noise term $\theta_3\delta_{ij}$ is crucial.

---

### From Q4: Probabilistic Inference (→ Our Exercise 3)

**Q4b (6 pts):** Derive that $p(\tau) = \int p(\tau|\theta)p(\theta)d\theta$ using the joint distribution and marginalization.

This directly relates to our **posterior predictive distribution** (Exercise 3c):

$$p(x^*|D) = \int p(x^*|\theta) p(\theta|D) d\theta$$

Same structure: integrate out the parameters.

> **Takeaway for our exam:** The posterior predictive is obtained by "marginalizing out" (integrating over) the parameters. This is exactly what makes Bayesian inference hard (Exercise 3d) — the integral is often intractable.

---

## Summary: What 1C Teaches Us for OUR Exam

| 1C Topic | Our Exercise | What It Reinforces |
|----------|-------------|-------------------|
| Bayes' theorem with concrete example | Ex 3a | Name all parts: posterior, likelihood, prior, evidence |
| Beta-Binomial conjugacy | Ex 3b | Concrete example of conjugate priors |
| Ridge regression = LSQ + $\lambda I$ | Ex 3/connections | MAP with Gaussian prior = ridge regression |
| GP positive definiteness, $O(N^3)$ | Ex 7 | Kernel matrix properties, computational cost |
| Kernel parameters → function properties | Ex 7b | Length-scale, signal variance, noise variance |
| Marginalization / posterior predictive | Ex 3c, 3d | Why Bayesian inference is hard: intractable integrals |
| Overfitting/underfitting | Ex 1b, general | Regularization, model complexity |

---

## Exam-Ready Insights from 1C

### 1. Concrete conjugate prior example (for Exercise 3b):
> "A Beta prior is conjugate to a Binomial likelihood: if $p(D|\theta) = \text{Binomial}$ and $p(\theta) = \text{Beta}(\alpha, \beta)$, then $p(\theta|D) = \text{Beta}(\alpha + \text{successes}, \beta + \text{failures})$. The posterior is the same family (Beta) as the prior."

### 2. Ridge regression connection (for Exercise 3 or "why" questions):
> "Adding regularization $\lambda\|w\|^2$ to the loss is equivalent to MAP estimation with a Gaussian prior $p(w) = \mathcal{N}(0, \frac{1}{\lambda}I)$ on the parameters."

### 3. GP key properties (for Exercise 7):
> - Covariance/kernel matrix must be **positive definite**
> - GP has $O(N^3)$ complexity (inverting $N \times N$ matrix)
> - Variance is low near observed data, high far from data
> - Kernel parameters control smoothness and amplitude
