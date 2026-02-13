# Lecture 05 — Probabilistic Model Fitting (Part 1)

**Source:** `slides/lect-05.txt` (97 slides)
**Exam mapping:** This is the **single most tested lecture**. Exercises 1, 2, and 3 (= 30 points) all draw from it.
**Priority:** CRITICAL

---

## Overview & Exam Mapping

| Slide Content                                                             | Maps to Exam Exercise                          | Points |
| ------------------------------------------------------------------------- | ---------------------------------------------- | ------ |
| What is ML, probabilistic model fitting idea                              | **Ex 1a** (What is ML?)                        | 5      |
| Log-likelihood as objective                                               | **Ex 1b** (What is a loss/objective function?) | 5      |
| Joint probability, log-likelihood, MLE, MAP formulas                      | **Ex 2a-d** (write down formulas)              | 10     |
| Bayes' theorem, conjugate priors, posterior predictive, why Bayes is hard | **Ex 3a-d** (Bayesian analysis)                | 10     |

---

## Exercise 1a: What is Machine Learning? (5 pts)

The prof's own definition (slide 9):

> **Machine learning** is the science of fitting mathematical models to data. We assume observations $(x_j, y_j) \in D$ were drawn from an unknown distribution. We decide for a parameterized probabilistic model $p(x, y \mid \theta)$ and fit it — i.e., we try to determine optimal parameters $\hat{\theta}$ by considering an appropriate objective and running an appropriate optimization procedure.

**What to write on the exam (5 pts):**

ML is the process of:

1. Assuming data come from an unknown distribution
2. Choosing a parameterized model $p(x \mid \theta)$
3. Finding optimal parameters $\hat{\theta}$ by optimizing an objective (e.g., maximizing likelihood, minimizing loss)
4. Using the fitted model for inference/prediction

**Key phrases to include:** "fit mathematical models to data," "parameterized model," "optimal parameters," "objective function."

---

## Exercise 1b: What is a Loss Function / Objective Function? (5 pts)

A **loss function** (also called objective function) is the quantity we optimize to find the best model parameters. It measures how well (or how poorly) the model fits the data.

- If we **minimize** it, it's called a **loss function** (measures badness of fit)
- If we **maximize** it, it's called an **objective function** (measures goodness of fit)

**Examples from this course:**

| Loss/Objective                | Formula                                                                | Minimize or Maximize? |
| ----------------------------- | ---------------------------------------------------------------------- | --------------------- |
| Squared loss (lect-02)        | $L(w) = \sum_{j=1}^{n}(y_j - f(x_j))^2$                                | Minimize              |
| Log-likelihood (this lecture) | $\mathcal{L}(\theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)$         | Maximize              |
| Negative log-likelihood       | $-\mathcal{L}(\theta)$                                                 | Minimize              |
| Log-posterior (MAP)           | $\log p(\theta \mid D) \propto \log p(D \mid \theta) + \log p(\theta)$ | Maximize              |

> **Exam tip:** The exam asks "What is a loss function? Describe its role in ML." Write the definition + the role: "A loss function quantifies the discrepancy between model predictions and observed data. In ML, we choose model parameters $\hat{\theta}$ that minimize (or maximize) this function. Different loss functions lead to different models (e.g., squared loss → regression, hinge loss → SVM, log-likelihood → MLE)."

---

## Exercise 2: Likelihood (10 pts total)

This exercise asks you to write down 4 formulas in sequence. Each builds on the previous one. **The exam gives you blanks to fill in — you must use the prof's exact notation.**

### The 4-Formula Chain (The Core of This Lecture)

---

### Ex 2a: Joint Probability for i.i.d. Data (2 pts)

**Concept:** If data points $x_1, \ldots, x_n$ are **independent and identically distributed** (i.i.d.) according to $p(x \mid \theta)$, then their joint probability is the **product** of individual probabilities.

**Why a product?** Independence means knowing one data point tells you nothing about another. The probability of seeing all of them together is the product of seeing each individually. "Identically distributed" means they all come from the same distribution $p(x \mid \theta)$ with the same parameters.

$$\boxed{p(D \mid \theta) = p(x_1, \ldots, x_n \mid \theta) = \prod_{j=1}^{n} p(x_j \mid \theta)}$$

> **This is the formula the exam asks you to fill in for Ex 2a. Get the product $\prod$ right — students who write $\sum$ here lose the full 2 points.**

**Where each symbol comes from:**

- $D = \{x_1, \ldots, x_n\}$ — the observed data (sample)
- $\theta$ — the unknown parameters of our probabilistic model
- $p(x_j \mid \theta)$ — the probability of observing data point $x_j$ under our model with parameters $\theta$
- $\prod_{j=1}^{n}$ — product over all $n$ data points (comes from the i.i.d. assumption)

---

### Ex 2b: Log-Likelihood Function (2 pts)

**Concept:** Products of many probabilities quickly become numerically tiny (underflow). Taking the logarithm converts the product into a **sum**, which is numerically stable and easier to differentiate.

Since $\log$ is a monotonically increasing function, maximizing $\log f(\theta)$ gives the same $\hat{\theta}$ as maximizing $f(\theta)$ — the argmax doesn't change.

The **log-likelihood function** is:

$$\boxed{\mathcal{L}(\theta) = \log L(\theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)}$$

> **This is the formula for Ex 2b. The key transformation: $\log$ turns $\prod$ into $\sum$. Write $\sum$ with $\log$ inside — not $\log$ of a $\sum$.**

**How we got here step by step:**

We start from the joint probability and apply $\log$:

$$\mathcal{L}(\theta) = \log p(D \mid \theta) = \log \prod_{j=1}^{n} p(x_j \mid \theta)$$

Applying the log rule $\log(a \cdot b) = \log a + \log b$ to the product:

$$= \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

**Notation warning (slide 30):** In this lecture, $\mathcal{L}$ denotes log-likelihood, NOT a loss function. The prof acknowledges "we are already running out of symbols." On the exam, context will make it clear.

---

### Ex 2c: Maximum Likelihood Estimator (3 pts)

**Concept:** The MLE is the value of $\theta$ that makes the observed data **most probable** under our model. We find it by maximizing the likelihood (or equivalently, the log-likelihood) over all possible $\theta$.

$$\boxed{\theta_{\text{ML}} = \arg\max_{\theta} \, p(D \mid \theta)}$$

Equivalently (since $\log$ is monotone):

$$\theta_{\text{ML}} = \arg\max_{\theta} \, \log p(D \mid \theta) = \arg\max_{\theta} \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

> **For Ex 2c, write the first form: $\theta_{\text{ML}} = \arg\max_{\theta} \, p(D \mid \theta)$. The exam gives you "θ_ML = " and you fill in the right side.**

**How to actually compute $\theta_{\text{ML}}$** (the meta-pattern):

1. **OBJECTIVE:** $\mathcal{L}(\theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)$
2. **CONSTRAINTS:** None (unconstrained optimization)
3. **DIFFERENTIATE:** $\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta} \overset{!}{=} 0$
4. **SOLVE:** the resulting equation for $\theta$

> **Connection:** This is the SAME meta-pattern you'll use in Ex 4 (argmin of distances), Ex 6 (SVM dual), and Ex 7 (GP kernel optimization). Learn it here, use it everywhere.

**Important subtlety (slide 45):** MLE finds $\theta$ that makes the **data** most likely given $\theta$:

$$\theta_{\text{ML}} = \arg\max_{\theta} \, p(D \mid \theta)$$

This is NOT the same as finding the most likely $\theta$ given the data. That would be:

$$\theta_{\text{MAP}} = \arg\max_{\theta} \, p(\theta \mid D)$$

MLE asks: "which $\theta$ makes the data probable?" MAP asks: "which $\theta$ is probable given the data?" MAP is "the right way around" — but requires more work (a prior).

---

### Ex 2d: Maximum A-Posteriori Estimator (3 pts)

**Concept:** Instead of asking "which $\theta$ makes the data likely?", MAP asks the better question: "which $\theta$ is most likely given the data?" This requires Bayes' theorem to "flip" the conditioning.

$$\boxed{\theta_{\text{MAP}} = \arg\max_{\theta} \, p(\theta \mid D)}$$

> **For Ex 2d, write exactly this: $\theta_{\text{MAP}} = \arg\max_{\theta} \, p(\theta \mid D)$. Notice the critical difference from MLE: $p(\theta \mid D)$ vs $p(D \mid \theta)$. The conditioning is flipped. Students who write $p(D \mid \theta)$ here get zero.**

**The relationship between MLE and MAP:**

Using Bayes' theorem: $p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$

Since $p(D)$ doesn't depend on $\theta$, it doesn't affect the argmax:

$$\theta_{\text{MAP}} = \arg\max_{\theta} \, p(D \mid \theta) \, p(\theta) = \arg\max_{\theta} \Big[\underbrace{\log p(D \mid \theta)}_{\text{log-likelihood}} + \underbrace{\log p(\theta)}_{\text{log-prior}}\Big]$$

So MAP = MLE + a prior term. The prior $p(\theta)$ acts as a **regularizer**: it penalizes parameter values we believe are unlikely before seeing data.

> **Critical connection to Ex 3:** MAP with a Gaussian prior $p(\theta) = \mathcal{N}(0, \frac{1}{\lambda}I)$ gives $\log p(\theta) = -\frac{\lambda}{2}\theta^\top\theta + \text{const}$, which is L2 regularization (ridge regression). **This connection is tested.** If the exam asks "what is the relationship between MAP and regularization?" — this is the answer.

---

## Exercise 3: Bayesian Data Analysis (10 pts total)

---

### Ex 3a: Bayes' Theorem (2 pts)

**Concept:** Bayes' theorem lets us compute the posterior $p(\theta \mid D)$ — the distribution over parameters _after_ seeing data — by combining the likelihood $p(D \mid \theta)$ with our prior beliefs $p(\theta)$.

Given a distribution $p(D \mid \theta)$, Bayes' theorem gives us:

$$\boxed{p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}}$$

**Name every part** (the exam may ask you to label them):

| Symbol             | Name                                  | Meaning                                                 |
| ------------------ | ------------------------------------- | ------------------------------------------------------- |
| $p(\theta \mid D)$ | **Posterior**                         | Our updated belief about $\theta$ after seeing data     |
| $p(D \mid \theta)$ | **Likelihood**                        | How probable the data is under parameter $\theta$       |
| $p(\theta)$        | **Prior**                             | Our initial belief about $\theta$ before seeing data    |
| $p(D)$             | **Evidence** (or marginal likelihood) | A normalizing constant so the posterior integrates to 1 |

The shorthand that the prof uses (slide 50):

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

Or in proportional form (dropping the evidence since it doesn't depend on $\theta$):

$$p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta)$$

> **Exam tip:** Both 1A and 1B ask this exact question. Write the full formula with the fraction, then label all four parts. That's the full 2 points.

---

### Ex 3b: Conjugate Priors (2 pts)

**Concept:** A prior $p(\theta)$ is called a **conjugate prior** for a likelihood $p(D \mid \theta)$ if the resulting posterior $p(\theta \mid D)$ belongs to the **same family** of distributions as the prior.

In other words: prior and posterior have the same functional form (e.g., both are Gaussian, or both are Gamma, or both are Beta). Only the parameters change.

**Why this matters:** If the posterior is the same type of distribution as the prior, we get a closed-form posterior — no intractable integrals needed. This is a huge computational convenience.

**The prof's terminology (slide 68):** When posterior and prior are from the same family:

- $p(\theta)$ is called a **conjugate prior** of $p(D \mid \theta)$
- $p(\theta \mid D)$ is called a **reproducing distribution**

**Concrete examples from this lecture:**

| Likelihood                                                                     | Conjugate Prior                                                          | Posterior                                          |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------- |
| Gaussian $\mathcal{N}(x \mid \mu, \sigma^2)$ (unknown $\mu$)                   | Gaussian $\mathcal{N}(\mu \mid \mu_0, \sigma_0^2)$                       | Gaussian $\mathcal{N}(\mu \mid \mu_n, \sigma_n^2)$ |
| Gaussian $\mathcal{N}(x \mid \mu, \lambda^{-1})$ (unknown precision $\lambda$) | Gamma $G(\lambda \mid \alpha_0, \beta_0)$                                | Gamma $G(\lambda \mid \alpha_n, \beta_n)$          |
| Gaussian (unknown $\mu$ and $\lambda$)                                         | Normal-Gamma $NG(\mu, \lambda \mid \mu_0, \lambda_0, \alpha_0, \beta_0)$ | Normal-Gamma                                       |
| Binomial (from Exam 1C)                                                        | Beta $\text{Beta}(\theta \mid \alpha, \beta)$                            | Beta $\text{Beta}(\theta \mid \alpha', \beta')$    |

> **What to write on the exam for Ex 3b:** "A conjugate prior is a prior distribution $p(\theta)$ such that the posterior $p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta)$ belongs to the same family of distributions as $p(\theta)$. For example, a Gaussian prior is conjugate to a Gaussian likelihood (for the mean parameter): if we start with a Gaussian prior, the posterior is also Gaussian. Conjugate priors are useful because they lead to closed-form posteriors."

> **If they ask for an example:** The Gaussian-Gaussian conjugacy from this lecture (prior $\mathcal{N}(\mu_0, \sigma_0^2)$ → posterior $\mathcal{N}(\mu_n, \sigma_n^2)$), OR the Beta-Binomial from Exam 1C (prior $\text{Beta}(\alpha, \beta)$ + Binomial likelihood → posterior $\text{Beta}(\alpha', \beta')$, e.g., testing whether a coin/die is fair).

---

### Ex 3c: Posterior Predictive Distribution (2 pts)

**Concept:** The posterior predictive distribution answers: "Given the data I've seen, what is the probability of a **new** observation $x$?" It integrates over **all possible parameter values**, weighted by how likely each parameter value is given the data.

$$\boxed{p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta}$$

**Where each part comes from:**

- $p(x \mid \theta)$ — the model: probability of new observation $x$ under parameter $\theta$
- $p(\theta \mid D)$ — the posterior: how likely is each $\theta$ given the data
- The integral — we marginalize out (average over) $\theta$

**How we derive it (slide 53):**

We start by asking for $p(x \mid D)$ and introduce $\theta$ as a latent variable:

$$p(x \mid D) = \int p(x, \theta \mid D) \, d\theta$$

We factor the joint using the product rule: $p(x, \theta \mid D) = p(x \mid \theta, D) \, p(\theta \mid D)$.

Then we use the reasonable assumption that $x$ is conditionally independent of $D$ given $\theta$ (i.e., if you know the parameters, the old data tells you nothing extra about the new point): $p(x \mid \theta, D) = p(x \mid \theta)$.

This gives us: $p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$.

> **For Ex 3c:** Write the formula $p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$ and explain: "The posterior predictive distribution gives the probability of a new observation $x$ given the training data $D$, by averaging the model predictions $p(x \mid \theta)$ over all parameter values weighted by their posterior probability $p(\theta \mid D)$."

The prof calls this the **"holy grail" of Bayesian data analysis** (slide 53).

**The MAP approximation:** If we can't compute the integral, we approximate:

$$p(x \mid D) \approx p(x \mid \theta_{\text{MAP}})$$

This replaces the full integration with a single point estimate — much simpler but throws away uncertainty information.

**Concrete result from this lecture (slide 83):** For a Gaussian model with Gaussian prior on $\mu$ (known $\sigma^2$):

$$p(x \mid D) = \mathcal{N}(x \mid \mu_n, \sigma^2 + \sigma_n^2)$$

The variance $\sigma^2 + \sigma_n^2$ is **larger** than $\sigma^2$ alone — the extra $\sigma_n^2$ accounts for our uncertainty about $\mu$. This is a key insight: the posterior predictive is wider than the model with known parameters, because we're uncertain about $\mu$.

---

### Ex 3d: Why is Bayesian Inference Hard? (4 pts)

**This is worth 4 points — the most of any sub-part in Ex 3. Give at least 2-3 clear reasons.**

**Reason 1: The evidence integral $p(D)$ is intractable.**

To compute the posterior via Bayes' theorem:

$$p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$$

we need the evidence (normalizing constant):

$$p(D) = \int p(D \mid \theta) \, p(\theta) \, d\theta$$

This integral is over the **entire parameter space**. For most models, it has no closed-form solution. High-dimensional parameter spaces make numerical integration exponentially expensive.

**Reason 2: The posterior predictive integral is intractable.**

Even if we somehow obtain $p(\theta \mid D)$, the posterior predictive:

$$p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$$

is itself another integral that usually has no closed form (slide 53: "for most distributions, the integral is **excruciatingly difficult** to solve").

**Reason 3: Conjugate priors are rare / limited.**

Conjugate priors give closed-form posteriors, but they only exist for specific likelihood-prior pairs. For complex models (neural networks, non-Gaussian data), no conjugate prior exists. We're forced to use approximate methods (MCMC, variational inference).

**Reason 4: Trade-off between model appropriateness and tractability (slide 11).**

The prof states: "we may face a trade-off between a model's appropriateness and tractability." Gaussian models are tractable but may not be appropriate. Complex models may be appropriate but intractable. In practice, people often choose Gaussian models "not because they are always appropriate but because they are tractable."

> **What to write on the exam for Ex 3d:**
>
> "Bayesian inference is hard because:
>
> 1. **Computing the evidence $p(D) = \int p(D \mid \theta) p(\theta) d\theta$ is intractable** for most models — the integral over the full parameter space has no closed form and is exponentially expensive to approximate numerically in high dimensions.
> 2. **The posterior predictive $p(x \mid D) = \int p(x \mid \theta) p(\theta \mid D) d\theta$ requires another intractable integral** — even if we have the posterior, computing predictions still requires integration.
> 3. **Conjugate priors (which give closed-form posteriors) exist only for simple models** — for complex/realistic models, we must resort to approximations (MCMC sampling, variational inference), which are computationally expensive and may not converge.
>
> **Example:** For a univariate Gaussian with Gaussian prior on $\mu$ (this lecture), everything works out in closed form. But for a multivariate Gaussian with unknown mean AND covariance, the computations become 'messy' (prof's word). For neural networks or mixture models, closed-form Bayesian inference is impossible."

---

## The Gaussian MLE Derivation (Background for Understanding)

This derivation illustrates the meta-pattern applied to a Gaussian model. The exam doesn't ask you to derive these MLEs, but understanding the steps helps you recognize the same pattern in Ex 4 and Ex 6.

### Setting

Data $D = \{x_1, \ldots, x_n\}$ where $x_j \in \mathbb{R}^m$, assumed i.i.d. from:

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^m \det(\Sigma)}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)\right)$$

### Step 1 — OBJECTIVE (write the log-likelihood):

$$\mathcal{L}(\mu, \Sigma) = -\frac{n}{2}\left(\log(2\pi)^m + \log\det(\Sigma)\right) - \frac{1}{2}\sum_{j=1}^{n}(x_j - \mu)^\top \Sigma^{-1}(x_j - \mu)$$

### Step 2 — No constraints (unconstrained)

### Step 3 — DIFFERENTIATE w.r.t. $\mu$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial \mu} = \sum_{j=1}^{n} \Sigma^{-1}(x_j - \mu) = \Sigma^{-1}\sum_{j=1}^{n}(x_j - \mu) \overset{!}{=} 0$$

Since $\Sigma^{-1}$ is invertible, we need $\sum_{j=1}^{n}(x_j - \mu) = 0$.

### Step 4 — SOLVE:

$$\sum_{j=1}^{n} x_j = \sum_{j=1}^{n} \mu = n\mu$$

$$\boxed{\mu_{\text{ML}} = \frac{1}{n}\sum_{j=1}^{n} x_j = \bar{x}}$$

The MLE for the mean is the **sample mean**. Similarly (derivation on slides 38-40):

$$\boxed{\Sigma_{\text{ML}} = \frac{1}{n}\sum_{j=1}^{n}(x_j - \mu_{\text{ML}})(x_j - \mu_{\text{ML}})^\top}$$

> **Note the $\frac{1}{n}$ vs $\frac{1}{n-1}$:** The MLE uses $\frac{1}{n}$ (population covariance). The unbiased estimator uses $\frac{1}{n-1}$ (sample covariance). The MLE **always underestimates variance** — this is a form of overfitting (slide 44). For large $n$, the difference is negligible.

> **Connection to Ex 4:** Exercise 4 asks you to compute $\hat{x} = \arg\min_x \sum \|x_j - x\|^2$. The answer is the sample mean — exactly the same result as $\mu_{\text{ML}}$ above. Same meta-pattern, same answer.

---

## The Bayesian Update for Gaussian Mean (Key Worked Example)

This is the concrete example the prof works through in detail (slides 61-83). While the exam doesn't ask you to reproduce this derivation, understanding it helps you answer Ex 3b (conjugate priors) and Ex 3c (posterior predictive) with concrete examples.

### Setting

- Model: $p(x) = \mathcal{N}(x \mid \mu, \sigma^2)$ with **known** $\sigma^2$, **unknown** $\mu$
- Prior on $\mu$: $p(\mu) = \mathcal{N}(\mu \mid \mu_0, \sigma_0^2)$

### Result

The posterior is also Gaussian (conjugate!):

$$p(\mu \mid D) = \mathcal{N}(\mu \mid \mu_n, \sigma_n^2)$$

where:

$$\mu_n = \frac{n\sigma_0^2}{n\sigma_0^2 + \sigma^2}\hat{\mu}_n + \frac{\sigma^2}{n\sigma_0^2 + \sigma^2}\mu_0$$

$$\sigma_n^2 = \frac{\sigma_0^2 \sigma^2}{n\sigma_0^2 + \sigma^2}$$

### Key Intuitions (exam-ready "reason about" answers)

**About $\mu_n$ (posterior mean):**

- It's a **weighted average** of the MLE $\hat{\mu}_n = \frac{1}{n}\sum x_j$ and the prior mean $\mu_0$
- With $n = 0$ data: $\mu_n = \mu_0$ (we rely entirely on the prior)
- As $n \to \infty$: $\mu_n \to \hat{\mu}_n$ (data overwhelms the prior)
- **The prior becomes less important as we see more data**

**About $\sigma_n^2$ (posterior variance):**

- It **decreases** as $n$ increases — each new data point reduces our uncertainty about $\mu$
- The posterior becomes "more sharply peaked" with more data
- If $\sigma_0^2 = 0$ (total certainty in prior): $\mu_n = \mu_0$ always — data is ignored
- If $\sigma_0^2 \gg \sigma^2$ (very uncertain prior): $\mu_n \approx \hat{\mu}_n$ — even a few data points dominate

**The posterior predictive (slide 83):**

$$p(x \mid D) = \mathcal{N}(x \mid \mu_n, \sigma^2 + \sigma_n^2)$$

The predictive variance $\sigma^2 + \sigma_n^2$ is always **larger** than the model variance $\sigma^2$ alone, because we're uncertain about $\mu$. As $n \to \infty$, $\sigma_n^2 \to 0$ and $p(x \mid D) \to \mathcal{N}(x \mid \hat{\mu}_n, \sigma^2)$.

---

## The MAP = Regularization Connection (Critical for Ex 3)

This connection appears across multiple lectures (lect-04, 05, 06) and is tested in multiple forms.

**Claim:** MAP estimation with a Gaussian prior on $\theta$ is equivalent to MLE with L2 regularization (ridge regression).

**Why:** The MAP objective is:

$$\theta_{\text{MAP}} = \arg\max_\theta \left[\log p(D \mid \theta) + \log p(\theta)\right]$$

If $p(\theta) = \mathcal{N}(0, \frac{1}{\lambda}I)$, then:

$$\log p(\theta) = -\frac{\lambda}{2}\theta^\top\theta + \text{const}$$

So the MAP objective becomes:

$$\theta_{\text{MAP}} = \arg\max_\theta \left[\underbrace{\log p(D \mid \theta)}_{\text{log-likelihood}} - \underbrace{\frac{\lambda}{2}\theta^\top\theta}_{\text{L2 penalty}}\right]$$

Equivalently (flipping max to min):

$$\theta_{\text{MAP}} = \arg\min_\theta \left[-\log p(D \mid \theta) + \frac{\lambda}{2}\theta^\top\theta\right]$$

This is exactly the regularized loss $L = L_{\text{data}} + L_r$ from lect-04, where $L_r = \frac{\lambda}{2}\theta^\top\theta = \frac{\lambda}{2}\|\theta\|^2$.

> **Key takeaway:** Regularization isn't just a "trick" — it has a probabilistic interpretation. Adding an L2 penalty is the same as assuming the parameters have a Gaussian prior centered at zero. This is why regularized models generalize better: the prior encodes a preference for "simple" (small-magnitude) parameters.

---

## What to Skip in This Lecture

- **The bivariate Gaussian example** (slides 5-16): Illustrative. Know that fitting a Gaussian means finding $\hat{\mu}$ and $\hat{\Sigma}$, but don't memorize the bivariate density formula.
- **The detailed algebra of completing the square** (slides 65-69): Understand the result, don't memorize the intermediate steps.
- **The Gamma and Normal-Gamma conjugacy details** (slides 85-93): Know they exist as conjugate prior examples. Don't memorize the Gamma posterior update formulas $\alpha_n, \beta_n$ — the exam asks about conjugate priors **conceptually**, not these specific update rules.
- **The Student's t-distribution result** (slide 95): One-liner — "when the posterior predictive isn't Gaussian, it can be a Student's t." Not tested.

---

## Connections to Other Exercises

| This Lecture Concept                                 | Feeds Into                                                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| MLE meta-pattern (objective → differentiate → solve) | **Ex 4** (sample mean via same pattern), **Ex 6** (SVM dual), **Ex 7** (GP kernel)                        |
| MAP = regularization                                 | **Ex 3d** (why regularization works), **Ex 6** (SVM primal has regularization term $\frac{1}{2}w^\top w$) |
| Log-likelihood as objective                          | **Ex 7** (GP log-marginal likelihood is the objective to optimize)                                        |
| Conjugate priors → closed-form posterior             | **Ex 3b** (definition), **Ex 7** (GP has closed-form posterior because of Gaussian-Gaussian conjugacy)    |
| Posterior predictive via integration                 | **Ex 3c** (formula), **Ex 7** (GP predictive distribution is a posterior predictive)                      |

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 05:

| #   | Formula                                                              | Maps to                        | Memorize?                                                                  |
| --- | -------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------- |
| 1   | $p(D \mid \theta) = \prod_{j=1}^{n} p(x_j \mid \theta)$              | **Ex 2a**                      | YES — write this verbatim                                                  |
| 2   | $\mathcal{L}(\theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)$       | **Ex 2b**                      | YES — write this verbatim                                                  |
| 3   | $\theta_{\text{ML}} = \arg\max_\theta p(D \mid \theta)$              | **Ex 2c**                      | YES — write this verbatim                                                  |
| 4   | $\theta_{\text{MAP}} = \arg\max_\theta p(\theta \mid D)$             | **Ex 2d**                      | YES — write this verbatim                                                  |
| 5   | $p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$      | **Ex 3a**                      | YES — write + label all 4 parts                                            |
| 6   | Conjugate prior definition                                           | **Ex 3b**                      | YES — definition + one example                                             |
| 7   | $p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$ | **Ex 3c**                      | YES — write + explain in words                                             |
| 8   | 3 reasons why Bayes is hard                                          | **Ex 3d**                      | YES — intractable integrals, rare conjugate priors, tractability trade-off |
| 9   | MAP with Gaussian prior = L2 regularization                          | **Ex 3** (connection question) | YES — state this if asked about MAP or regularization                      |

### Verdict:

**Drill these formulas until you can write them blind.** This lecture alone covers 30 points of the exam. Formulas 1-4 are fill-in-the-blank questions — if you can write them from memory, that's 10 guaranteed points. Formulas 5-8 cover another 10 points. Formula 9 is a connection that could appear in any "explain" or "reason about" question.
