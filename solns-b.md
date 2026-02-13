# POML Exam 1B — Full Solutions

**Source:** `past-questions/POML_2324_WS_1B.txt`
**Target:** 70/80 points

---

## Task 1: Definitions

### a) What is machine learning? And why is it useful?

> **Question:** What is machine learning? And why is it useful?

**Machine learning is the science of fitting mathematical models to data.**

More precisely: ML is to run computer algorithms on exemplary data to adjust the parameters of other computer algorithms (i.e. models) so that these become able to perform cognitive tasks.

**Why is it useful?**
It allows us to solve problems where:

1.  The problem is too complex to program by hand (e.g., face recognition).
2.  The solution changes over time (e.g., spam filtering).
3.  We want to discover patterns in large datasets (data mining).

> **Exam note:** The definition "fitting mathematical models to data" is key. For utility, mention "solving tasks too complex for manual programming."

---

### b) What is a Loss function (or an objective function) and for what is it used in machine learning?

> **Question:** What is a Loss function (or an objective function) and for what is it used in machine learning?

A **loss function** (or objective function) $L(\theta)$ is a mathematical function that measures how well a model with parameters $\theta$ fits the given data.

- A **loss function** quantifies the discrepancy between the model's predictions and the observed data. The goal is to **minimize** the loss.
- An **objective function** is the general term for the function we optimize. It may include the loss plus regularization terms.

**Used for:**
It defines the goal of learning. The training process becomes an optimization problem: finding the parameters $\hat{\theta}$ that minimize the loss function.

$$\hat{\theta} = \arg\min_{\theta} L(\theta)$$

---

## Task 2: Probability

### a) joint probability for i.i.d.

> **Question:** joint probability for i.i.d.

Given a sample $D = \{x_1, \ldots, x_n\}$ drawn i.i.d. (independent and identically distributed) from $p(x \mid \theta)$:

$$p(D \mid \theta) = \prod_{j=1}^{n} p(x_j \mid \theta)$$

> **Exam note:** Write the product symbol $\prod$. "i.i.d." implies the joint distribution factorizes.

---

### b) Log likelihood with respect to theta

> **Question:** Log likelihood with respect to theta

The log-likelihood function is:

$$\mathcal{L}(\theta) = \log p(D \mid \theta) = \log \prod_{j=1}^{n} p(x_j \mid \theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

> **Why we Use Logs:** Numerical stability (avoids underflow of small probabilities) and easier differentiation (turns products into sums).

---

### c) ML - estimatar

> **Question:** ML - estimatar

The **Maximum Likelihood Estimator (MLE)** is:

$$\theta_{\text{ML}} = \arg\max_{\theta} p(D \mid \theta) = \arg\max_{\theta} \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

It finds the parameter value that makes the observed data most probable.

---

### d) MAP- estimator

> **Question:** MAP- estimator

The **Maximum A-Posteriori (MAP) Estimator** is:

$$\theta_{\text{MAP}} = \arg\max_{\theta} p(\theta \mid D)$$

Using Bayes' theorem (and ignoring the constant denominator $p(D)$):

$$\theta_{\text{MAP}} = \arg\max_{\theta} \left[ p(D \mid \theta) \cdot p(\theta) \right] = \arg\max_{\theta} \left[ \log p(D \mid \theta) + \log p(\theta) \right]$$

It maximizes the posterior probability, incorporating prior beliefs $p(\theta)$.

---

## Task 3: Bayesian inference

### a) b) c) what is the posterior predictive distribution ? Definition or mathematical formulation

> **Question:** what is the posterior predictive distribution ? Definition or mathematical formulation

The **posterior predictive distribution** is the probability distribution of a new data point $x$, given the observed training data $D$, properly accounting for uncertainty in the model parameters $\theta$.

**Mathematical Formulation:**
We compute it by **marginalizing out** the parameters $\theta$ using the posterior distribution:

$$p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$$

> **Exam note:** "Marginalizing out" or "integrating over" $\theta$ are the key phrases.

---

### What is a conjugate distribution?

> **Question:** What is a conjugate distribution?

A prior $p(\theta)$ is a **conjugate prior** regarding a likelihood function $p(D \mid \theta)$ if the resulting posterior distribution $p(\theta \mid D)$ is in the **same family of distributions** as the prior.

**Example:**

- Prior: Gaussian $\mathcal{N}(\mu_0, \sigma_0^2)$
- Likelihood: Gaussian $\mathcal{N}(x \mid \mu, \sigma^2)$
- Posterior: Gaussian $\mathcal{N}(\mu_n, \sigma_n^2)$

---

### What are problems during Bayesian Inference?

> **Question:** What are problems during Bayesian Inference?

**The main problem is computational tractability of the integrals.**

Specifically:

1.  **Computing the Evidence $p(D)$:** The denominator in Bayes' theorem requires integrating over the entire parameter space $\theta$:
    $$p(D) = \int p(D \mid \theta) \, p(\theta) \, d\theta$$
    This integral is usually intractable (no closed-form solution exists).
2.  **High Dimensions:** As the dimension of $\theta$ increases, numerical approximation methods (like grid search or simple Monte Carlo) fail due to the "curse of dimensionality."

3.  **Complex Models:** Closed-form solutions (conjugate priors) only exist for simple models (exponential family). For modern complex models (e.g., Deep Learning), we cannot use exact inference and must rely on approximations like MCMC or Variational Inference.

> **Exam note:** "Intractable integrals" is the keyword. Mentioning "evidence" and "high dimensionality" ensures full points.

---

### d) Bayes Theorem

> **Question:** Bayes Theorem

$$p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$$

**Where:**

- $p(\theta \mid D)$: **Posterior** (belief after seeing data)
- $p(D \mid \theta)$: **Likelihood** (data fit)
- $p(\theta)$: **Prior** (initial belief)
- $p(D)$: **Evidence** (normalization constant)

---

## Task 4: Convex Approximation

### Setup

> **Question:** $x^ = \arg\min_x \sum \|x_j - x\|$

_(Note: The protocol likely meant squared Euclidean distance, which is standard for this problem)._

$$\hat{x} = \arg\min_{x} \sum_{j=1}^{n} \|x_j - x\|^2$$

---

### a) what would be way to minimize this problem?

> **Question:** what would be way to minimize this problem?

This is an unconstrained convex optimization problem. We solve it by:

1.  Writing down the objective function $L(x)$.
2.  Computing the gradient $\nabla_x L(x)$.
3.  Setting the gradient to zero (necessary condition for optimum).
4.  Solving the resulting linear equation for $x$.

---

### b) Close form for w^

> **Question:** Close form for $w^\hat{}$

_(Assuming "w^" refers to the optimal $\hat{x}$)._

We solve this using the **Meta-Pattern**:

**Step 1 — OBJECTIVE:**
The objective function to minimize is:
$$L(x) = \sum_{j=1}^n \|x_j - x\|^2 = \sum_{j=1}^n (x_j - x)^\top (x_j - x)$$

**Step 2 — CONSTRAINTS:**
There are no constraints, so no Lagrangian is needed.

**Step 3 — DIFFERENTIATE:**
First, expand the term inside the sum:
$$(x_j - x)^\top (x_j - x) = x_j^\top x_j - 2x_j^\top x + x^\top x$$

Substitute back into $L(x)$ and differentiate with respect to $x$:
$$\nabla_x L(x) = \sum_{j=1}^n \nabla_x (x_j^\top x_j - 2x_j^\top x + x^\top x)$$

Using matrix calculus rules $\nabla_x (b^\top x) = b$ and $\nabla_x (x^\top x) = 2x$:
$$\nabla_x L(x) = \sum_{j=1}^n (0 - 2x_j + 2x)$$
$$= -2\sum_{j=1}^n x_j + \sum_{j=1}^n 2x$$
$$= -2\sum_{j=1}^n x_j + 2n x$$

**Step 4 — SOLVE:**
Set the gradient to zero:
$$-2\sum_{j=1}^n x_j + 2n \hat{x} \overset{!}{=} 0$$
$$2n \hat{x} = 2\sum_{j=1}^n x_j$$

Dividing by $2n$:
$$\boxed{\hat{x} = \frac{1}{n} \sum_{j=1}^n x_j}$$

---

### c) Interpretation for w

> **Question:** Interpretation for $w$

The optimal vector $\hat{x}$ is satisfying the least squares criterion is simply the **sample mean** (centroid) of the data points $x_1, \ldots, x_n$.

It corresponds to the **Maximum Likelihood Estimator (MLE)** for the mean $\mu$ of a Gaussian distribution $\mathcal{N}(\mu, I)$ given the data.

---

## Task 5: Kernel trick

### a) b) what is the kernel trick?

> **Question:** what is the kernel trick?

The kernel trick is a method to apply linear algorithms to non-linear problems.

**Two-step definition:**

1.  Rewrite the algorithm such that input vectors $x$ only appear in the form of **dot products** $\langle x_i, x_j \rangle$.
2.  Replace these dot products with a **kernel function** $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$, which computes the dot product in a (possibly high-dimensional) feature space.

---

### Why is it interesting for machine learning?

> **Question:** Why is it interesting for machine learning?

1.  **Non-linearity:** It allows linear classifiers (like SVMs) or regressors (like Ridge Regression) to learn **non-linear boundaries** by implicitly mapping data to a high-dimensional space.
2.  **Computational Efficiency:** We compute $k(x_i, x_j)$ directly in the input space without ever explicitly computing the high-dimensional coordinates $\phi(x)$, avoiding the "curse of dimensionality" in calculation.

---

## Task 6: SVM

### a) b) c) write down the dual form of L_2 SVM?

> **Question:** write down the dual form of L_2 SVM?

The **Dual Problem of L2 SVM training** is:

$$\boxed{\arg\min_{\mu \in \Delta_{n-1}} \; \mu^\top \left[ Z^\top Z + yy^\top + \frac{1}{C} I \right] \mu}$$

**Where:**

- $\Delta_{n-1}$ is the **simplex**:
  $$\text{s.t.} \quad \sum_{j=1}^{n} \mu_j = 1, \quad \mu_j \geq 0$$
- $Z$ is the matrix with columns $z_j = y_j x_j$.
- $C$ is the regularization constant (from the primal).

> **Exam note:** This specific matrix form ($Z^\top Z + yy^\top + \frac{1}{C}I$) on the simplex is the one taught in this course (Bauckhage). Memorize it exactly. It combines the kernel matrix ($Z^\top Z$), the bias term effect ($yy^\top$), and the L2 slack penalty ($\frac{1}{C}I$).

---

### What is a way to solve is?

> **Question:** What is a way to solve is?

The **Frank-Wolfe Algorithm** (or Conditional Gradient Method).

It is an iterative algorithm for constrained optimization:

1.  Linearize the objective function at the current estimate.
2.  Find the minimizer of the linear function over the constraint set (simplex).
3.  Move towards that minimizer.

---

### What are the solutions for w and b?

> **Question:** What are the solutions for w and b?

From the KKT conditions:

$$\boxed{w = \sum_{j=1}^{n} \mu_j y_j x_j = Z\mu}$$

$$\boxed{b = -\sum_{j=1}^{n} \mu_j y_j = -y^\top \mu}$$

---

## Task 7: Gaussian Processes

### a) b) when we have input y that is not 0, how can we still solve (exercise sheet)

> **Question:** when we have input y that is not 0, how can we still solve (exercise sheet)

_(Question likely means "when y does not have zero mean")._

GPs assume a prior mean of zero, i.e., $y \sim \mathcal{N}(0, C)$. If the data $y$ has an arbitrary non-zero mean:

1.  **Center the training data:** Compute the sample mean $\bar{y} = \frac{1}{n}\sum y_i$ and subtract it: $\tilde{y}_i = y_i - \bar{y}$.
2.  **Train/Predict:** Perform GP inference using the centered targets $\tilde{y}$.
3.  **Add Mean Back:** For prediction outputs, add the mean back: $\hat{f}(x_*) = \text{GP\_predict}(\tilde{y}) + \bar{y}$.

---

### What could be a loss function for this? How could you derive a solution for this non-convex loss function? Write down a way to solve this.

> **Question:** What could be a loss function for this? How could you derive a solution for this non-convex loss function? Write down a way to solve this.

**1. Loss Function:**
The objective is to maximize the **log-marginal likelihood**. Equivalently, we minimize the **negative log-likelihood**:

$$\mathcal{L}(\theta) = -\log p(y \mid X, \theta) = \frac{1}{2} \log \det(C_\theta) + \frac{1}{2} y^\top C_\theta^{-1} y + \text{const}$$

We want to find hyperparameters $\theta$ that **minimize** this loss.

**2. Solution Derivation (Meta-Pattern):**

- **Objective:** $\mathcal{L}(\theta)$ (as above).
- **Differentiate:** Compute gradients with respect to hyperparameters $\theta_j$:
  $$\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{1}{2} \text{tr}\left(C^{-1} \frac{\partial C}{\partial \theta_j}\right) - \frac{1}{2} y^\top C^{-1} \frac{\partial C}{\partial \theta_j} C^{-1} y$$
- **Solve:** The equation $\nabla_\theta \mathcal{L} = 0$ is non-linear and has no closed-form solution.

**3. How to Solve (Algorithm):**
We use **Gradient Descent** (or Ascent on likelihood).
Since the objective is **non-convex** (possibility of local optima), we must use:
**Gradient Descent with Multiple Random Restarts.**

> **Exam note:** The key keywords are "Gradient Descent" and "Multiple Random Restarts" because of "Non-convexity".
