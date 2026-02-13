# POML Exam 1A — Full Solutions

**Source:** `past-questions/POML_2324_WS_1A.txt`
**Target:** 70/80 points

---

## Exercise 1: Definitions (5+5 = 10 P)

### a) What is Machine Learning?

> **Question:** What is Machine Learning?

**Machine learning is the science of fitting mathematical models to data.**

More precisely: ML is to run computer algorithms on exemplary data to adjust the parameters of other computer algorithms (i.e. models) so that these become able to perform cognitive tasks.

> **Exam note:** The prof uses this exact phrasing on slide 6 of lect-01. Reproduce it verbatim. The key phrase is "fitting mathematical models to data." Writing just this sentence likely earns full marks.

---

### b) What is a loss function / objective function? Describe their roles in Machine Learning.

> **Question:** What is a loss function / objective function? Describe their roles in Machine Learning?

A **loss function** (or objective function) $L(\theta)$ is a mathematical function that measures how well a model with parameters $\theta$ fits the given data.

- A **loss function** quantifies the discrepancy between the model's predictions and the observed data. The goal is to **minimize** the loss.
- An **objective function** is the general term for the function we optimize (minimize or maximize). It may include the loss plus additional terms like regularization.

**Role in ML:** The entire process of "fitting a model to data" is formulated as an optimization problem:

$$\hat{\theta} = \arg\min_{\theta} L(\theta)$$

We choose a parameterized model $p(x \mid \theta)$ or $f(x \mid w)$, define a loss function that measures how poorly the current parameters explain the data, and then run an optimization procedure (MLE, MAP, gradient descent, ...) to find the parameters that minimize this loss.

**Examples the prof uses:**

- Least squares loss: $L(w) = \|\Phi^\top w - y\|^2$
- Negative log-likelihood: $L(\theta) = -\sum_{j=1}^{n} \log p(x_j \mid \theta)$
- SVM loss: $L(w, \xi) = \frac{1}{2} w^\top w + C \sum_j \xi_j^2$

> **Exam note:** Give the definition + at least one concrete example. Mentioning both "measures model fit" and "we minimize/maximize it" covers the key points.

---

## Exercise 2: Likelihood (2+2+3+3 = 10 P)

### a) Joint probability of an i.i.d. sample (2 P)

> **Question:** Given a sample $D= \{x_1,...,x_n\}$, express their joint probability: $P(D \mid \theta) = p(x_1,...,x_n \mid \Theta) =$ ?

Given a sample $D = \{x_1, \ldots, x_n\}$ drawn i.i.d. from $p(x \mid \theta)$:

$$p(D \mid \theta) = p(x_1, \ldots, x_n \mid \theta) = \prod_{j=1}^{n} p(x_j \mid \theta)$$

> **Why it factors as a product:** Independence means the joint probability equals the product of the individual probabilities. Identical distribution means each factor uses the same $p(\cdot \mid \theta)$.

---

### b) Log-likelihood function (2 P)

> **Question:** Using the above, write down the log-likelihood function $L(\theta) =$ ?

Products of many probabilities quickly become numerically tiny (underflow). Taking the logarithm converts the product into a **sum**, which is numerically stable and easier to differentiate. Since $\log$ is a monotonically increasing function, maximizing $\log f(\theta)$ gives the same $\hat{\theta}$ as maximizing $f(\theta)$ — the argmax doesn't change.

We start from the joint probability and apply $\log$:

$$\mathcal{L}(\theta) = \log p(D \mid \theta) = \log \prod_{j=1}^{n} p(x_j \mid \theta)$$

Applying the log rule $\log(a \cdot b) = \log a + \log b$ to the product:

$$= \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

The **log-likelihood function** is:

$$\boxed{\mathcal{L}(\theta) = \log L(\theta) = \sum_{j=1}^{n} \log p(x_j \mid \theta)}$$

> **This is the formula for Ex 2b. The key transformation: $\log$ turns $\prod$ into $\sum$. Write $\sum$ with $\log$ inside — not $\log$ of a $\sum$.**

---

### c) Maximum Likelihood Estimator (3 P)

> **Question:** Write down a maximum likelihood estimator of $\theta$, $\Theta_{ML} =$ ?

The MLE is the value of $\theta$ that makes the observed data **most probable** under our model. We find it by maximizing the likelihood (or equivalently, the log-likelihood) over all possible $\theta$.

$$\theta_{\text{ML}} = \arg\max_{\theta} \, p(D \mid \theta)$$

Equivalently (since $\log$ is monotonous):

$$\theta_{\text{ML}} = \arg\max_{\theta} \, \log p(D \mid \theta) = \arg\max_{\theta} \sum_{j=1}^{n} \log p(x_j \mid \theta)$$

---

### d) Maximum A-Posteriori Estimator (3 P)

> **Question:** Write down a maximum a-posteriori estimator for $\theta$, $\Theta_{MAP} =$ ?

$$\theta_{\text{MAP}} = \arg\max_{\theta} \, p(\theta \mid D)$$

> **What this means:** MAP finds the parameter $\hat{\theta}$ that is **most probable given the data**. Unlike MLE which maximizes $p(D \mid \theta)$, MAP maximizes $p(\theta \mid D)$ — the posterior probability of the parameters given what we observed. This requires Bayes' theorem to "flip" the conditioning.

Using Bayes' theorem: $p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$

Since $p(D)$ doesn't depend on $\theta$, it doesn't affect the argmax:

$$\theta_{\text{MAP}} = \arg\max_{\theta} \, p(D \mid \theta) \, p(\theta) = \arg\max_{\theta} \Big[\underbrace{\log p(D \mid \theta)}_{\text{log-likelihood}} + \underbrace{\log p(\theta)}_{\text{log-prior}}\Big]$$

So MAP = MLE + a prior term. The prior $p(\theta)$ acts as a **regularizer**: it penalizes parameter values we believe are unlikely before seeing data.

---

## Exercise 3: Bayesian Data Analysis (2+2+2+4 = 10 P)

### a) Bayes' theorem for the posterior (2 P)

> **Question:** Given a distribution $p(D \mid \theta)$. Write down Bayes’ theorem to compute the posterior.

Given a distribution $p(D \mid \theta)$, the posterior is:

$$p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$$

where:

- $p(\theta \mid D)$ is the **posterior** — our updated belief about $\theta$ after seeing data
- $p(D \mid \theta)$ is the **likelihood** — probability of the data given the parameters
- $p(\theta)$ is the **prior** — our initial belief about $\theta$ before seeing data
- $p(D)$ is the **evidence** (or marginal likelihood) — a normalizing constant

The prof's shorthand:

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

> **Exam note:** The exam says "write down Bayes' theorem to compute the posterior." Write the formula AND name all four parts. That's likely worth the full 2 points.

---

### b) Explain the notion of conjugate priors (2 P)

> **Question:** Explain the notion of conjugate priors.

A prior $p(\theta)$ is called a **conjugate prior** for a likelihood $p(D \mid \theta)$ if the resulting posterior $p(\theta \mid D)$ belongs to the **same family of distributions** as the prior.

In other words: if prior and posterior are from the same distribution family, we say they are **conjugate distributions**, and the prior is a conjugate prior of the likelihood.

**Why this matters:** Conjugate priors are a mathematical convenience — they lead to closed-form solutions for the posterior, avoiding the intractable integral in the denominator of Bayes' theorem. People like them because they "simplify" computations.

| Likelihood                                                                     | Conjugate Prior                                                          | Posterior                                          |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------- |
| Gaussian $\mathcal{N}(x \mid \mu, \sigma^2)$ (unknown $\mu$)                   | Gaussian $\mathcal{N}(\mu \mid \mu_0, \sigma_0^2)$                       | Gaussian $\mathcal{N}(\mu \mid \mu_n, \sigma_n^2)$ |
| Gaussian $\mathcal{N}(x \mid \mu, \lambda^{-1})$ (unknown precision $\lambda$) | Gamma $G(\lambda \mid \alpha_0, \beta_0)$                                | Gamma $G(\lambda \mid \alpha_n, \beta_n)$          |
| Gaussian (unknown $\mu$ and $\lambda$)                                         | Normal-Gamma $NG(\mu, \lambda \mid \mu_0, \lambda_0, \alpha_0, \beta_0)$ | Normal-Gamma                                       |
| Binomial (from Exam 1C)                                                        | Beta $\text{Beta}(\theta \mid \alpha, \beta)$                            | Beta $\text{Beta}(\theta \mid \alpha', \beta')$    |

---

### c) Posterior predictive distribution (2 P)

> **Question:** Explain the notion of posterior predictive distribution and write down the posterior predictive distribution based on the notation in subtask a).

The **posterior predictive distribution** answers the question: _"What is the probability of observing a new data point $x$ given the training data $D$?"_

It is computed by **marginalizing out** the parameters $\theta$ using the posterior:

$$p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$$

> **What this means intuitively:** Instead of committing to a single "best" parameter (like MLE or MAP), we average predictions over **all possible** parameter values, weighted by how probable each value is given the data. This accounts for our uncertainty about $\theta$.

> **Connection to MAP (partial credit):** If the integral is too hard, we can approximate $p(x \mid D) \approx p(x \mid \theta_{\text{MAP}})$. This is cheaper but ignores parameter uncertainty.

> **Exam note:** Write both (1) the definition/explanation and (2) the formula. The formula alone is worth points even if the explanation is weak.

---

### d) Why is Bayesian inference often very hard in practice? (4 P)

> **Question:** Reason about why Bayesian inference is often very hard in practise. Give at least one example.

**Main reason: The integral is intractable.**

The posterior predictive requires computing:

$$p(x \mid D) = \int p(x \mid \theta) \, p(\theta \mid D) \, d\theta$$

and the posterior itself requires computing the evidence:

$$p(D) = \int p(D \mid \theta) \, p(\theta) \, d\theta$$

These integrals are **analytically intractable** for most non-trivial models. They can only be solved in closed form for very special cases (e.g., conjugate prior families like Gaussian-Gaussian).

**Specific reasons (give at least 2-3 of these):**

1. **High-dimensional integration:** When $\theta$ is high-dimensional, the integrals become exponentially expensive to compute or approximate numerically. There is no general closed-form solution.

2. **Computing the evidence $p(D)$:** The denominator of Bayes' theorem requires integrating the likelihood times the prior over the entire parameter space. For complex models, this is intractable.

3. **Non-conjugate models:** If the prior is not conjugate to the likelihood, the posterior does not have a known closed form, and we must resort to expensive approximation methods (MCMC sampling, variational inference).

4. **Trade-off between appropriateness and tractability:** The prof emphasizes that we often face a choice between models that are appropriate (but intractable) and models that are tractable (but potentially inappropriate). Gaussian models are popular not because they are always appropriate, but because they are tractable.

**Example:** For a neural network with millions of parameters, computing the full posterior $p(\theta \mid D)$ is completely infeasible — we cannot integrate over millions of dimensions analytically.

> **Exam note:** This is a 4-point "reason about why" question. Give **at least 3 distinct arguments** with keywords: intractable integrals, high dimensionality, evidence computation, non-conjugate priors, tractability vs. appropriateness trade-off. Each argument is worth ~1-1.5 points.

---

## Exercise 4: Simple Convex Optimization (4+4+2 = 10 P)

### Setup

We are given:

$$\hat{x} = \arg\min_{x} \sum_{j=1}^{n} \|x_j - x\|^2$$

where $x_j, x \in \mathbb{R}^m$.

---

### a) How would you solve this? (4 P)

> **Question:** Consider $\hat{x} = \arg\min_x \sum \|x_j - x\|^2$. How would you solve this?

This is an **unconstrained optimization problem**. We apply the meta-pattern:

**Step 1 — OBJECTIVE:** The objective function is:

$$L(x) = \sum_{j=1}^{n} \|x_j - x\|^2 = \sum_{j=1}^{n} (x_j - x)^\top (x_j - x)$$

**Step 2 — CONSTRAINTS:** There are no constraints, so no Lagrangian is needed.

**Step 3 — DIFFERENTIATE:**

We take the derivative with respect to $x$. First, expand the term inside the sum:

$$(x_j - x)^\top (x_j - x) = x_j^\top x_j - x_j^\top x - x^\top x_j + x^\top x$$

Using $a^\top b = b^\top a$, we have $x^\top x_j = x_j^\top x$, so:

$$= x_j^\top x_j - 2x_j^\top x + x^\top x$$

Now, substitute this back into $L(x)$ and differentiate term by term:

$$L(x) = \sum_{j=1}^{n} (x_j^\top x_j - 2x_j^\top x + x^\top x)$$

Using matrix calculus rules $\nabla_x (b^\top x) = b$ and $\nabla_x (x^\top x) = 2x$:

$$\frac{\partial L}{\partial x} = \sum_{j=1}^{n} (0 - 2x_j + 2x) = -2\sum_{j=1}^{n} x_j + \sum_{j=1}^{n} 2x$$

Note that $\sum_{j=1}^{n} 2x = 2n x$ (summing a constant $n$ times).

$$\frac{\partial L}{\partial x} = -2\sum_{j=1}^{n} x_j + 2n x$$

**Step 4 — SOLVE:** Set the derivative to zero and solve for $\hat{x}$:

$$-2\sum_{j=1}^{n} x_j + 2n \hat{x} \overset{!}{=} 0$$

$$2n \hat{x} = 2\sum_{j=1}^{n} x_j$$

Dividing by $2n$:

$$\boxed{\hat{x} = \frac{1}{n} \sum_{j=1}^{n} x_j}$$

> **Exam note:** Even just writing these 4 steps (objective, no constraints, differentiate, solve) earns partial credit. The prof follows this skeleton in every derivation.

---

### b) Explicitly compute $\hat{x}$ (4 P)

> **Question:** Explicitly compute $\hat{x}$?

**Step 3 in detail — DIFFERENTIATE:**

We expand the objective:

$$L(x) = \sum_{j=1}^{n} (x_j - x)^\top (x_j - x)$$

Expanding each term:

$$(x_j - x)^\top(x_j - x) = x_j^\top x_j - x_j^\top x - x^\top x_j + x^\top x = x_j^\top x_j - 2 x_j^\top x + x^\top x$$

> **Rule used:** For vectors $a, b$: $a^\top b = b^\top a$ (scalar), so $x_j^\top x = x^\top x_j$.

Substituting back:

$$L(x) = \sum_{j=1}^{n} \left( x_j^\top x_j - 2 x_j^\top x + x^\top x \right)$$

Now differentiate with respect to $x$:

> **Matrix calculus rules used:**
>
> - $\frac{\partial}{\partial x}(a^\top x) = a$
> - $\frac{\partial}{\partial x}(x^\top x) = 2x$
> - Constants (not involving $x$) differentiate to zero.

$$\frac{\partial L}{\partial x} = \sum_{j=1}^{n} \left( 0 - 2 x_j + 2x \right) = -2 \sum_{j=1}^{n} x_j + 2n \cdot x$$

> **Step shown explicitly:** The sum $\sum_{j=1}^{n} 2x = 2x + 2x + \cdots + 2x = 2n \cdot x$ (summing the constant $2x$ exactly $n$ times).

**Step 4 — SOLVE:** Set equal to zero:

$$-2 \sum_{j=1}^{n} x_j + 2n \cdot x = 0$$

$$2n \cdot x = 2 \sum_{j=1}^{n} x_j$$

$$\boxed{\hat{x} = \frac{1}{n} \sum_{j=1}^{n} x_j}$$

---

### c) What does your computation in b) imply? (2 P)

> **Question:** What does your computation in b) imply?

The result $\hat{x} = \frac{1}{n} \sum_{j=1}^{n} x_j$ is the **sample mean** (arithmetic average) of the data.

This implies that **the point that minimizes the sum of squared distances to all data points is the sample mean**. In other words:

- The sample mean is the optimal "representative" of the data in the least-squares sense.
- This is exactly the **MLE of the mean** of a Gaussian distribution: $\mu_{\text{ML}} = \frac{1}{n} \sum_{j=1}^{n} x_j$.
- This also explains why the **k-means centroid update** computes the mean of all points assigned to a cluster — it minimizes the within-cluster sum of squared distances.

> **Exam note:** The key connection to state: minimizing sum of squared distances yields the sample mean, which is also the MLE of $\mu$ for a Gaussian. Mentioning both earns full marks.

---

## Exercise 5: The Kernel Trick (5+5 = 10 P)

### a) What is the kernel trick? (5 P)

> **Question:** What’s the kernel trick?

The kernel trick is a two-step procedure:

**Step 1:** Rewrite an algorithm for data analysis in such a way that all input data **only appears in form of inner products** with other data, i.e., expressions of the form $x_i^\top x_j$.

**Step 2:** Replace every occurrence of such inner products by evaluations of a **Mercer kernel function**: $x_i^\top x_j \longrightarrow k(x_i, x_j)$.

**Why this works:** A Mercer kernel $k(x, y)$ is a positive semidefinite function for which there exists a (possibly high- or infinite-dimensional) feature map $\phi: \mathbb{R}^m \to \mathbb{R}^M$ such that:

$$k(x, y) = \phi(x)^\top \phi(y)$$

This means the kernel **implicitly computes inner products** not in the original data space $\mathbb{R}^m$ but in a latent feature space $\mathbb{R}^M$ (where often $M \gg m$ or even $M = \infty$).

**The key benefit:** We can use **linear** models to solve **non-linear** problems, because data that is not linearly separable in $\mathbb{R}^m$ may become linearly separable in $\mathbb{R}^M$ — and we never need to explicitly compute $\phi(x)$, only the kernel $k(x, y)$.

> **Exam note:** The prof defines this on lect-11, slide 55. Reproduce the two steps and mention Mercer's theorem. The two-step definition is the core answer.

---

### b) Where and how is the kernel trick used in Machine Learning? (5 P)

> **Question:** Where and how is the Kernel trick used in Machine Learning?

The kernel trick is used in virtually every fundamental ML method. Key examples:

**1. Support Vector Machines (kernel SVMs):**

- Replace the Gram matrix $X^\top X$ (with entries $x_i^\top x_j$) by a kernel matrix $K$ (with entries $k(x_i, x_j)$).
- For L2 SVMs: $X^\top X \odot yy^\top + yy^\top + \frac{1}{C}I \longrightarrow K \odot yy^\top + yy^\top + \frac{1}{C}I$
- During application: $x^\top X \longrightarrow k(x)^\top$ where $[k(x)]_j = k(x, x_j)$.
- This allows SVMs to learn non-linear decision boundaries.

**2. Kernel Least Squares Regression:**

- The dual LSQ model $y = \phi_x^\top \Phi (\Phi^\top \Phi)^{-1} y$ becomes $y = k_x^\top K^{-1} y$.
- With regularization: $y = k_x^\top (K + \lambda I)^{-1} y$.

**3. Gaussian Process Regression:**

- GP regression is essentially kernelized regression. The predictive mean is:
  $\mu_* = K_{*x}(K_{xx} + \sigma^2 I)^{-1} y$
- The kernel function defines the covariance structure of the GP, controlling the smoothness and properties of the functions it can represent.

**4. Other methods:** Kernel PCA, kernel k-means, kernel LDA — all use the same principle.

**Common kernels:**

| Kernel               | Formula                                                    | Feature space dim $M$ |
| -------------------- | ---------------------------------------------------------- | --------------------- |
| Linear               | $k(x,y) = x^\top y$                                        | $m$                   |
| Inhomogeneous linear | $k(x,y) = x^\top y + b$                                    | $m + 1$               |
| Polynomial           | $k(x,y) = (x^\top y + b)^d$                                | $\binom{m+d}{d}$      |
| Gaussian (RBF)       | $k(x,y) = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$ | $\infty$              |

> **Exam note:** Give at least 2-3 concrete examples of where the kernel trick is used. Mentioning SVMs + GP regression + one more is safe for full marks. Include at least one kernel formula.

---

## Exercise 6: Support Vector Machines (5+5+5 = 15 P)

### a) Write down the (formal) Dual Problem of L2 SVM training (5 P)

> **Question:** Write down the (formal) Dual Problem of L2 SVM training.

First, recall the notation. Given training data $D = \{(x_j, y_j)\}_{j=1}^n$ with $x_j \in \mathbb{R}^m$ and $y_j \in \{-1, +1\}$, define:

$$z_j = y_j x_j, \quad Z = \begin{pmatrix} | & | & & | \\ z_1 & z_2 & \cdots & z_n \\ | & | & & | \end{pmatrix} \in \mathbb{R}^{m \times n}$$

The **dual problem of L2 SVM training** is:

$$\boxed{\arg\min_{\mu \in \Delta_{n-1}} \; \mu^\top \left[ Z^\top Z + yy^\top + \frac{1}{C} I \right] \mu}$$

where $\Delta_{n-1}$ is the standard $(n-1)$-simplex, meaning:

$$\text{s.t.} \quad 1^\top \mu = 1, \quad \mu \geq 0$$

**Equivalently written out:**

$$\arg\min_{\mu} \; \mu^\top \left[ Z^\top Z + yy^\top + \frac{1}{C} I \right] \mu$$
$$\text{s.t.} \quad \sum_{j=1}^{n} \mu_j = 1, \quad \mu_j \geq 0 \; \forall j$$

> **Where this comes from (the derivation, for understanding):**
>
> The primal problem is:
> $$\arg\min_{w, b, \rho, \xi} \frac{1}{2}(w^\top w + b^2 + C\xi^\top\xi) - \rho \quad \text{s.t.} \quad Z^\top w - by \geq \rho \mathbf{1} - \xi$$
>
> We form the Lagrangian and apply KKT conditions:
>
> - $\frac{\partial L}{\partial w} = 0 \Rightarrow w = Z\mu$
> - $\frac{\partial L}{\partial b} = 0 \Rightarrow b = -y^\top\mu$
> - $\frac{\partial L}{\partial \xi} = 0 \Rightarrow \xi = \frac{1}{C}\mu$
> - $\frac{\partial L}{\partial \rho} = 0 \Rightarrow 1^\top\mu = 1$
>
> Substituting back eliminates $w, b, \rho, \xi$ and yields the dual above.

> **Exam note:** This is worth 5 points. Writing the dual formula with both constraints ($1^\top \mu = 1$, $\mu \geq 0$) and the correct matrix $Z^\top Z + yy^\top + \frac{1}{C}I$ is the full answer.

---

### b) How would you solve this problem? (5 P)

> **Question:** How would you solve this problem?

This is a **quadratic program on the simplex** $\Delta_{n-1}$ (minimize a quadratic function subject to $1^\top \mu = 1$, $\mu \geq 0$).

**Solution method: The Frank-Wolfe algorithm** (also known as the conditional gradient method).

The Frank-Wolfe algorithm is an iterative procedure:

1. **Initialize:** $\mu^{(0)} = \frac{1}{n} \mathbf{1}$ (uniform distribution on the simplex)
2. **For** $t = 0, 1, 2, \ldots, T$:
   - Compute the gradient: $g = 2\left[Z^\top Z + yy^\top + \frac{1}{C}I\right] \mu^{(t)}$
   - Find the simplex vertex that minimizes the linear approximation: $s = e_{\arg\min_j g_j}$ (i.e., the standard basis vector corresponding to the smallest gradient component)
   - Set the step size: $\beta = \frac{2}{t+2}$
   - Update: $\mu^{(t+1)} = \mu^{(t)} + \beta(s - \mu^{(t)})$
3. **Return** $\mu^{(T)}$

**Key properties:**

- Each iterate stays on the simplex (convex combination of current point and a vertex)
- The algorithm linearizes the objective at each step and moves toward the minimizer of the linear approximation on the simplex
- Converges for convex objectives, which the L2 SVM dual is (since $Z^\top Z + yy^\top + \frac{1}{C}I$ is positive definite)

> **Exam note:** Name the Frank-Wolfe algorithm and describe its key idea (iterative, linearize objective, step toward simplex vertex). Even naming the algorithm + "it iteratively solves quadratic programs on the simplex" is worth significant partial credit.

---

### c) Recovering $w$ and $b$ from KKT conditions (5 P)

> **Question:** Using the Karush-Kuhn-Tucker conditions: Once you have solved the L2 training problem, how can $w$ and $b$ be computed? $w=$? $b=$?

Once we have solved the dual and obtained $\hat{\mu}$, the KKT conditions give us:

$$\boxed{w = Z\hat{\mu} = \sum_{j=1}^{n} \hat{\mu}_j y_j x_j}$$

$$\boxed{b = -y^\top \hat{\mu} = -\sum_{j=1}^{n} \hat{\mu}_j y_j}$$

> **Where these come from:**
>
> From KKT condition 1: $\frac{\partial L}{\partial w} = w - Z\mu = 0 \Rightarrow w = Z\mu$
>
> Recall $Z = X \cdot \text{diag}(y)$, so $z_j = y_j x_j$, hence $w = \sum_j \mu_j y_j x_j$.
>
> From KKT condition 2: $\frac{\partial L}{\partial b} = b + y^\top \mu = 0 \Rightarrow b = -y^\top \mu$

> **Note on support vectors:** For data points $x_j$ that are **not** support vectors, $\mu_j = 0$. So the sums above effectively only run over the support vectors (where $\mu_j > 0$). This makes application of the trained SVM efficient.

> **Exam note:** These two formulas are the core answer. Write them clearly. The derivation from KKT earns extra points but even just the formulas should get most of the 5 marks.

---

## Exercise 7: Gaussian Processes (5+10 = 15 P)

### a) Non-zero mean: How to work with arbitrary $y$? (5 P)

> **Question:** Given $x=$ input, $y=$ output, $N(0,C(\Theta))$. We often model with $N(0,C(\theta))$, but you cannot always assume a zero-mean. How do you work with arbitrary $y$?

The standard GP model assumes $y \sim \mathcal{N}(0, C(\theta))$, i.e., a **zero-mean** Gaussian process.

But in practice, the target vector $y$ may not have zero mean. To handle this:

**Step 1 — Center the data:** Compute the sample mean:

$$\bar{y} = \frac{1}{n} \sum_{j=1}^{n} y_j$$

**Step 2 — Subtract the mean:** Work with the centered targets:

$$\tilde{y} = y - \bar{y} \cdot \mathbf{1}$$

Now $\tilde{y}$ has (approximately) zero mean, and we can apply the standard zero-mean GP model:

$$\tilde{y} \sim \mathcal{N}(0, C(\theta))$$

**Step 3 — Add the mean back for predictions:** When making predictions for test inputs $x_*$, compute the GP prediction using $\tilde{y}$ and then add back $\bar{y}$:

$$\hat{y}_* = k_*^\top (K_{xx} + \sigma^2 I)^{-1} \tilde{y} + \bar{y}$$

> **In summary:** Center $y$ by subtracting its mean, fit the zero-mean GP to the centered data, then shift predictions back by adding the mean.

> **Exam note:** The key answer is: subtract the sample mean from $y$, apply the zero-mean GP, add the mean back to predictions. This is a "how would you handle" question — stating the procedure clearly is worth the full 5 points.

---

### b) Optimal kernel parameters: formulas and optimization (10 P)

> **Question:** How can you get the optimal parameters for $C(\Theta)$? The explanation should include the kernel for $C(\Theta)$ as well as other explicit formulas. How would you solve the optimisation problem, considering it is not convex?

We model $y \sim \mathcal{N}(0, C(\theta))$ where $C(\theta) = K(\theta) + \sigma^2 I$ is a parameterized covariance matrix.

A common kernel for GP regression is:

$$[C(\theta)]_{ij} = \theta_0 \exp\!\left(-\frac{(x_i - x_j)^2}{2\theta_1^2}\right) + \theta_2 \, x_i x_j + \theta_3^2 \, \delta_{ij}$$

where $\theta = (\theta_0, \theta_1, \theta_2, \theta_3)^\top$ are the **hyperparameters** and $\delta_{ij}$ is the Kronecker delta (so $\theta_3^2 \delta_{ij}$ represents the noise variance $\sigma^2$).

**Step 1 — OBJECTIVE (log-likelihood):**

Given the model $y \sim \mathcal{N}(0, C(\theta))$, the log-likelihood of the hyperparameters is:

$$\boxed{\mathcal{L}(\theta) = \log p(y \mid \theta) = -\frac{1}{2} \log \det C - \frac{1}{2} y^\top C^{-1} y + \text{const}}$$

We want to find:

$$\theta_{\text{ML}} = \arg\max_{\theta} \, \mathcal{L}(\theta)$$

**Step 2 — DIFFERENTIATE (gradient):**

The gradient of the log-likelihood with respect to each hyperparameter $\theta_l$ is:

$$\frac{\partial \mathcal{L}}{\partial \theta_l} = -\frac{1}{2} \operatorname{tr}\!\left(C^{-1} \frac{\partial C}{\partial \theta_l}\right) + \frac{1}{2} y^\top C^{-1} \frac{\partial C}{\partial \theta_l} C^{-1} y$$

**Step 3 — SOLVE (gradient ascent):**

There is **no closed-form solution** for $\theta_{\text{ML}}$. We must use iterative gradient-based optimization:

1. Guess initial parameters $\theta^{(0)}$
2. For $t = 0, 1, 2, \ldots$:
   $$\theta^{(t+1)} \leftarrow \theta^{(t)} + \eta_t \cdot \nabla_\theta \mathcal{L}$$
3. Stop when converged

**Why this is hard — the optimization is NOT convex:**

The log-likelihood $\mathcal{L}(\theta)$ is generally a **non-convex** function of $\theta$. This means:

- Gradient ascent may converge to a **local** maximum, not the global one
- The result depends on the initial guess $\theta^{(0)}$

**Solution to non-convexity: Gradient ascent (or descent) with multiple random restarts.**

- Run the gradient optimization procedure multiple times, each time starting from a **different random initialization** $\theta^{(0)}$
- Keep the result that achieves the highest log-likelihood
- This increases the chance of finding a good (ideally global) optimum by exploring different regions of the parameter space

> **Additional detail:** We may also need to enforce non-negativity of parameters ($\theta_0, \theta_1, \theta_2, \theta_3 > 0$), which makes this a constrained optimization problem. The computational cost of GP training is dominated by inverting the $n \times n$ covariance matrix $C$, which is $O(n^3)$.

> **Exam note:** This is worth 10 points — the single highest-value sub-question. To earn full marks, you need ALL of:
>
> 1. The log-likelihood formula (3-4 pts)
> 2. The gradient formula (2-3 pts)
> 3. "Use gradient ascent" (1-2 pts)
> 4. "Non-convex → multiple random restarts" (2-3 pts)
>
> Even writing just the log-likelihood formula and saying "gradient ascent with multiple random restarts because non-convex" earns significant partial credit.

---

## Quick Reference: Key Formulas for Exam 1A

| Exercise | Formula                                                                                   | Points |
| -------- | ----------------------------------------------------------------------------------------- | ------ |
| Ex 2a    | $p(D \mid \theta) = \prod_{j=1}^n p(x_j \mid \theta)$                                     | 2      |
| Ex 2b    | $\mathcal{L}(\theta) = \sum_{j=1}^n \log p(x_j \mid \theta)$                              | 2      |
| Ex 2c    | $\theta_{\text{ML}} = \arg\max_\theta p(D \mid \theta)$                                   | 3      |
| Ex 2d    | $\theta_{\text{MAP}} = \arg\max_\theta p(\theta \mid D)$                                  | 3      |
| Ex 3a    | $p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)}$                              | 2      |
| Ex 4b    | $\hat{x} = \frac{1}{n}\sum_{j=1}^n x_j$                                                   | 4      |
| Ex 6a    | $\min_{\mu \in \Delta} \mu^\top[Z^\top Z + yy^\top + \frac{1}{C}I]\mu$                    | 5      |
| Ex 6c    | $w = Z\hat{\mu}, \quad b = -y^\top\hat{\mu}$                                              | 5      |
| Ex 7b    | $\mathcal{L}(\theta) = -\frac{1}{2}\log\det C - \frac{1}{2}y^\top C^{-1}y + \text{const}$ | 10     |
