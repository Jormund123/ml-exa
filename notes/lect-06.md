# Lecture 06 — Probabilistic Model Fitting (Part 2)

**Source:** `slides/lect-06.txt` (87 slides)
**Exam mapping:** Deepens **Exercise 3** (especially 3b and 3d). Concretely proves the **MAP = regularization** connection. Establishes the **least squares = MLE** equivalence (background for Ex 1b).
**Priority:** HIGH — the (dis)advantages list is directly tested in Ex 3d (4 pts), and the MAP regression formula is the concrete proof of the regularization connection.

---

## Overview & Exam Mapping

| Slide Content | Maps to Exam Exercise | Priority |
|---|---|---|
| Recap of lect-05 formulas (slides 4-11) | Ex 2, 3 | Already covered in lect-05 notes |
| Multivariate Gaussian + Normal-Wishart (slides 13-20) | Background only | LOW — skip details |
| **(Dis)advantages of Bayesian estimation (slides 21-24)** | **Ex 3b, 3d** | **CRITICAL — memorize these bullet points** |
| Least squares = MLE with Gaussian noise (slides 25-39) | Ex 1b (connection) | MEDIUM |
| **Bayesian regression: MAP = regularization (slides 43-52)** | **Ex 3 (connection question)** | **CRITICAL** |
| Posterior predictive for regression (slides 52-55) | Ex 3c, Ex 7 | HIGH |
| Logistic regression = cross entropy (slides 59-76) | Not tested directly | LOW |
| tanh neuron = Bayesian classifier (slides 77-84) | Not tested | SKIP |

---

## (Dis)advantages of Bayesian Parameter Estimation (→ Ex 3b, 3d)

**This is the prof's own bullet-point list (slides 22-24). The exam's Ex 3d asks you to "reason about why Bayesian inference is often very hard." These bullet points ARE the answer.**

### Advantages (+)

**1. The prior $p(\theta)$ allows incorporating domain-specific knowledge.**

Before seeing data, we can encode what we already know about reasonable parameter values. For example, if we know weights should be small, we use a prior centered at zero.

**2. Conjugate priors lead to closed-form solutions.**

If $p(\theta)$ is a conjugate prior of the likelihood $p(D \mid \theta)$, then:
- The posterior $p(\theta \mid D)$ has a closed-form expression
- The evidence $p(D) = \int p(D \mid \theta) p(\theta) \, d\theta$ has a closed-form solution
- The posterior predictive $p(x \mid D) = \int p(x \mid \theta) p(\theta \mid D) \, d\theta$ has a closed-form solution

All densities in the **exponential family** come with conjugate priors (slide 22).

**3. Posterior predictive distributions can be very valuable (slide 23).**

They give us not just a point prediction but a full distribution over predictions — we know *how uncertain* we are. This is crucial for safety-critical applications and is the foundation for Gaussian processes (→ Ex 7).

### Disadvantages (−)

> **These are the answers for Ex 3d (4 pts). Memorize all four.**

**4. Representing prior knowledge via (conjugate) prior distributions is "anything but intuitive" (slide 23).**

The prof's exact words: representing prior knowledge in terms of conjugate prior distributions is "by and large a fantasy of proponents of the Bayesian framework." Concretely: what are "good" initial guesses for the parameters of a Gamma prior? Where do they come from? How are they justified?

> **Exam-ready phrasing:** "Choosing meaningful prior distributions is difficult in practice. Conjugate priors are chosen for mathematical convenience, not because they necessarily reflect genuine prior knowledge."

**5. Bayesian statistics does not allow for falsifying hypotheses (slide 24).**

There is no "false" hypothesis — only different degrees of belief. If beliefs stem from mathematical convenience rather than genuine knowledge, the conclusions may be questionable.

**6. If priors are not conjugate, computing the evidence is intractable (slide 24).**

$$p(D) = \int p(D \mid \theta) \, p(\theta) \, d\theta$$

has **no closed-form solution** for non-conjugate priors. This integral is over the full parameter space → curse of dimensionality. Must resort to approximate methods like **Markov Chain Monte Carlo (MCMC)**.

**7. "Real world" Bayesian inference is often very, very involved (slide 86).**

The prof explicitly states this in the summary. Even for the "simple" examples in this lecture, the algebra was "tedious yet straightforward." For complex models, it becomes intractable.

> **What to write on the exam for Ex 3d (4 pts):**
>
> "Bayesian inference is hard because:
> 1. **The evidence integral $p(D) = \int p(D|\theta)p(\theta)d\theta$ is intractable** for non-conjugate priors — no closed-form solution exists, and numerical integration suffers from the curse of dimensionality.
> 2. **Choosing meaningful priors is difficult** — conjugate priors are chosen for mathematical convenience, not because they reflect real prior knowledge. For example, what are 'good' initial parameters for a Gamma prior?
> 3. **The posterior predictive $p(x|D) = \int p(x|\theta)p(\theta|D)d\theta$ requires yet another integral** that is usually intractable.
>
> **Example:** For a univariate Gaussian with a Gaussian prior on $\mu$ (lect-05), everything works in closed form. But for complex models (neural networks, mixture models), neither the evidence nor the posterior have closed forms, requiring expensive approximate methods like MCMC."

---

## Least Squares = MLE (→ Ex 1b connection)

This section (slides 25-39) proves a beautiful equivalence. While the exam doesn't ask you to derive this, the **conclusion** is important for understanding what least squares regression really does.

### Setting

Training data $D = \{(x_j, y_j)\}_{j=1}^n$, linear model:

$$y_j = \phi_j^\top w + \epsilon_j$$

where $\phi_j = \begin{bmatrix} 1 \\ x_j \end{bmatrix}$ is the feature vector (includes bias term).

**Least squares approach:** Minimize $\|\Phi^\top w - y\|^2$ → gives $\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y$.

### The MLE Approach (slides 30-36)

If we assume i.i.d. Gaussian noise $\epsilon_j \sim \mathcal{N}(0, \sigma^2)$, then each observation is:

$$y_j \sim \mathcal{N}(y_j \mid \phi_j^\top w, \sigma^2)$$

The log-likelihood for $w$ is:

$$\mathcal{L}(w \mid D) = -n\log\sqrt{2\pi\sigma^2} - \frac{1}{2\sigma^2}\sum_{j=1}^{n}(\phi_j^\top w - y_j)^2$$

To maximize $\mathcal{L}(w)$, we need to minimize the second term (the first doesn't depend on $w$). The constant $\frac{1}{2\sigma^2}$ doesn't affect the argmin, so:

$$w_{\text{ML}} = \arg\min_w \sum_{j=1}^{n}(\phi_j^\top w - y_j)^2 = \arg\min_w \|\Phi^\top w - y\|^2$$

### The Key Equivalence

$$\boxed{\text{Least squares regression} \iff \text{Maximum likelihood with Gaussian noise model}}$$

> **Why this matters for Ex 1b:** If asked "what is a loss function?", you can now say: "The squared loss $\|\Phi^\top w - y\|^2$ arises naturally as the negative log-likelihood under a Gaussian noise assumption. Different noise assumptions lead to different loss functions."

### Bonus Result: Squared Error = Scaled Variance (slide 39)

The MLE for $\sigma^2$ turns out to be:

$$\sigma^2_{\text{ML}} = \frac{1}{n}\|\Phi^\top w_{\text{ML}} - y\|^2$$

This means the squared error $\|\Phi^\top w - y\|^2 = n\sigma^2$ is a **scaled variance**. So least squares minimizes a variance.

---

## Bayesian Regression: The MAP = Regularization Proof (→ Ex 3, CRITICAL)

This is the **concrete derivation** of the connection stated abstractly in lect-05 notes. The exam can ask about this connection in Ex 3 (as a "reason about" question) or as part of a ridge regression question.

### Setting (slides 43-47)

Same regression model as above, but now we do Bayesian estimation:
- **Model:** $y_j \sim \mathcal{N}(\phi_j^\top w, \sigma^2)$ with **known** $\sigma^2$, **unknown** $w$
- **Prior on $w$:** $p(w) = \mathcal{N}(w \mid 0, \sigma_0^2 I)$ — a Gaussian centered at zero

The prior says: "before seeing data, we believe the weights $w$ are small (centered at zero) with uncertainty $\sigma_0^2$ per component."

### The Posterior (slides 47-50)

We compute $p(w \mid D) \propto p(D \mid w) \, p(w)$:

$$p(w \mid D) \propto \exp\left[-\frac{1}{2\sigma^2}\|\Phi^\top w - y\|^2\right] \cdot \exp\left[-\frac{1}{2\sigma_0^2}\|w\|^2\right]$$

Combining the exponents:

$$p(w \mid D) \propto \exp\left[-\frac{1}{2}\left(\frac{1}{\sigma^2}\|\Phi^\top w - y\|^2 + \frac{1}{\sigma_0^2}\|w\|^2\right)\right]$$

This is an exponential of a quadratic in $w$ → the posterior is Gaussian (conjugate!):

$$p(w \mid D) = \mathcal{N}(w \mid \mu_n, \Sigma_n)$$

where:

$$\Sigma_n = \left[\frac{1}{\sigma^2}\Phi\Phi^\top + \frac{1}{\sigma_0^2}I\right]^{-1}$$

$$\mu_n = \frac{1}{\sigma^2}\Sigma_n \Phi y$$

### The MAP Estimator (slides 50-52)

Since the mode of a Gaussian equals its mean: $w_{\text{MAP}} = \mu_n$.

After simplification (multiply through by $\sigma^2$):

$$\boxed{w_{\text{MAP}} = \left[\Phi\Phi^\top + \frac{\sigma^2}{\sigma_0^2}I\right]^{-1}\Phi y}$$

### Side-by-Side Comparison (slide 52)

$$w_{\text{ML}} = (\Phi\Phi^\top)^{-1}\Phi y$$

$$w_{\text{MAP}} = \left(\Phi\Phi^\top + \underbrace{\frac{\sigma^2}{\sigma_0^2}}_{\lambda}I\right)^{-1}\Phi y$$

> **This is the formula the exam might ask you to compare. The only difference is the $\lambda I$ term added to $\Phi\Phi^\top$.**

### The Connection Made Explicit

Setting $\lambda = \frac{\sigma^2}{\sigma_0^2}$, the MAP estimator becomes:

$$w_{\text{MAP}} = (\Phi\Phi^\top + \lambda I)^{-1}\Phi y$$

This is **exactly the ridge regression** (L2-regularized least squares) solution! The MAP objective is:

$$w_{\text{MAP}} = \arg\min_w \left[\underbrace{\|\Phi^\top w - y\|^2}_{\text{data fit (squared loss)}} + \underbrace{\lambda \|w\|^2}_{\text{regularization (from prior)}}\right]$$

$$\boxed{\text{MAP with Gaussian prior } \mathcal{N}(0, \sigma_0^2 I) \iff \text{Ridge regression with } \lambda = \frac{\sigma^2}{\sigma_0^2}}$$

> **Exam 1C cross-reference:** Exam 1C (Q2b-c) asks to derive the ridge regression solution $\mu = (K^\top K + \lambda I)^{-1}K^\top g$. This is the same formula in different notation. The connection MAP = regularization appears across all ML exams.

### Interpreting $\lambda = \frac{\sigma^2}{\sigma_0^2}$

| Scenario | Effect |
|----------|--------|
| $\sigma_0^2 \to \infty$ (very uncertain prior, "I know nothing about $w$") | $\lambda \to 0$ → no regularization → $w_{\text{MAP}} \to w_{\text{ML}}$ |
| $\sigma_0^2 \to 0$ (very confident prior, "I know $w \approx 0$") | $\lambda \to \infty$ → strong regularization → $w_{\text{MAP}} \to 0$ |
| $\sigma^2$ large (noisy data) | $\lambda$ large → trust the prior more than the data |
| $\sigma^2$ small (clean data) | $\lambda$ small → trust the data more than the prior |

> **Key insight:** Regularization strength $\lambda$ encodes the **ratio of data noise to prior uncertainty**. This is not an arbitrary hyperparameter — it has a probabilistic meaning.

---

## Posterior Predictive for Bayesian Regression (→ Ex 3c, Ex 7)

Given the posterior $p(w \mid D) = \mathcal{N}(w \mid \mu_n, \Sigma_n)$, the posterior predictive distribution for a new input $x$ is (slide 53-54):

$$p(y \mid x, D) = \int \underbrace{\mathcal{N}(y \mid \phi_x^\top w, \sigma^2)}_{\text{model}} \cdot \underbrace{\mathcal{N}(w \mid \mu_n, \Sigma_n)}_{\text{posterior}} \, dw$$

This integral of two Gaussians yields another Gaussian:

$$\boxed{p(y \mid x, D) = \mathcal{N}(y \mid \mu_x, \sigma_x^2)}$$

where:

$$\mu_x = \phi_x^\top \mu_n$$

$$\sigma_x^2 = \sigma^2 + \phi_x^\top \Sigma_n \phi_x$$

**What each part means:**
- $\mu_x = \phi_x^\top \mu_n$ — the predicted mean equals our MAP model's prediction
- $\sigma_x^2 = \sigma^2 + \phi_x^\top \Sigma_n \phi_x$ — the predictive variance has **two sources**:
  - $\sigma^2$ — inherent noise in the data (irreducible)
  - $\phi_x^\top \Sigma_n \phi_x$ — **uncertainty in $w$** (reducible with more data)

> **Connection to Ex 7 (Gaussian Processes):** This posterior predictive is exactly what a GP computes! A GP generalizes this idea: instead of being uncertain about weight vector $w$, we're uncertain about the entire function. The GP predictive also has the form "mean prediction ± uncertainty," where uncertainty is low near training data and high far from it. **The Bayesian regression posterior predictive is a preview of GP regression.**

**Key property (from the illustrations, slides 57-58):**
- Where data is dense → $\Sigma_n$ is small → $\sigma_x^2 \approx \sigma^2$ → tight confidence bands
- Where data is missing → $\Sigma_n$ is large → $\sigma_x^2 \gg \sigma^2$ → wide confidence bands
- With outliers → Bayesian regression is more "robust" than plain least squares

The prof calls this "fairly robust or confidence-scored explanations of the given data" (slide 56).

---

## Logistic Regression = Cross Entropy (→ Low Priority)

Slides 59-76 show that the logistic loss is actually a sum of cross entropies in disguise. This is a conceptual insight, not tested on our exam.

**One-line summary:** The logistic loss $L = \sum_j \log(1 + e^{-y_j a_j})$ can be rewritten as a sum of cross entropies $H(z_j, q_j) = -\sum_k (z_j)_k \log(q_j)_k$ between target distributions $z_j$ and predicted distributions $q_j = \sigma(a_j)$.

> **Exam relevance: LOW.** Know that "logistic loss = cross entropy" in case a "what is" question appears. Don't memorize the derivation.

---

## tanh Neuron = Bayesian Decision Maker (→ Skip)

Slides 77-84 show that a tanh neuron computes the difference of two Bayesian posteriors under symmetric Gaussian assumptions. This is an elegant result but has **zero exam points**.

> **Skip entirely.** Not tested in 1A, 1B, or 1C.

---

## What to Skip in This Lecture

- **Normal-Wishart conjugacy details** (slides 13-20) — Know that the Normal-Wishart is the conjugate prior for a multivariate Gaussian with unknown $\mu$ and $\Sigma$. Don't memorize the Wishart distribution formula or the update equations $\kappa_n, \nu_n, W_n$.
- **The logistic loss → cross entropy derivation** (slides 59-76) — Conceptual, not tested.
- **The tanh = Bayesian classifier derivation** (slides 77-84) — Not tested at all.
- **Detailed algebra of the Bayesian regression posterior** (slides 47-49) — Understand the result, don't memorize the completing-the-square steps.

---

## Connections to Other Exercises

| This Lecture Concept | Feeds Into |
|---------------------|------------|
| (Dis)advantages of Bayesian estimation | **Ex 3d** (4 pts — "why is Bayes hard?") — use the bullet points from slides 22-24 |
| Least squares = MLE | **Ex 1b** — "loss function has a probabilistic interpretation" |
| $w_{\text{MAP}} = (\Phi\Phi^\top + \lambda I)^{-1}\Phi y$ | **Ex 3** (MAP = regularization), **Ex 6** (SVM regularization term $\frac{1}{2}w^\top w$) |
| Posterior predictive $p(y \mid x, D)$ with uncertainty bands | **Ex 3c** (write down posterior predictive), **Ex 7** (GP is the generalization of this) |
| "Variance is large where data is missing" | **Ex 7** (GP property: high uncertainty far from training points) |

---

## End-of-Lecture Summary

### Exam-relevant content from Lecture 06:

| # | Content | Maps to | Memorize? |
|---|---------|---------|-----------|
| 1 | (Dis)advantages of Bayesian estimation (4 bullet points) | **Ex 3d** (4 pts) | **YES — memorize all four disadvantages** |
| 2 | Conjugate priors: "mathematical convenience," "lead to closed-form solutions," "all exponential family distributions have them" | **Ex 3b** (2 pts) | YES — supplements lect-05 answer |
| 3 | Least squares $\iff$ MLE with Gaussian noise | **Ex 1b** (connection) | Know the equivalence, don't memorize the proof |
| 4 | $w_{\text{MAP}} = (\Phi\Phi^\top + \lambda I)^{-1}\Phi y$ where $\lambda = \sigma^2/\sigma_0^2$ | **Ex 3** (MAP = regularization) | **YES — this is the concrete formula** |
| 5 | MAP with Gaussian prior = ridge regression (L2 regularization) | **Ex 3** | **YES — state this connection if asked** |
| 6 | Posterior predictive: $p(y \mid x, D) = \mathcal{N}(\mu_x, \sigma^2 + \phi_x^\top\Sigma_n\phi_x)$ | **Ex 3c**, **Ex 7** | Know the structure (mean + increased variance) |
| 7 | "Variance is large where data is sparse" | **Ex 7** | YES — GP property |

### Verdict:

**Drill the (dis)advantages list and the MAP = regularization formula.** These directly answer Ex 3b (2 pts) and Ex 3d (4 pts). The rest deepens understanding but doesn't introduce new exam formulas beyond what lect-05 already provides.

The two lectures together (lect-05 + lect-06) form a complete unit for Exercises 1, 2, and 3 = **30 points**. After studying both, you should be able to:
- Write all 4 formulas in Ex 2 from memory ✓ (lect-05)
- Write Bayes' theorem and label all parts ✓ (lect-05)
- Explain conjugate priors with an example ✓ (lect-05 + lect-06)
- Write the posterior predictive formula ✓ (lect-05)
- Give 3+ reasons why Bayes is hard ✓ (lect-06)
- State the MAP = regularization connection ✓ (lect-06)
