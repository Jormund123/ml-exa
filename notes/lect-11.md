# Lecture 11: Support Vector Machines (Part 2) — Exam Notes

> **Exam mapping:** This single lecture feeds **Exercise 5** (kernel trick, 10 pts) and **Exercise 6** (L2 SVM dual, 15 pts) = **25 points total**. This is the highest-ROI lecture in the course.

---

## Table of Contents

1. [Notation Reference](#1-notation-reference)
2. [Three SVM Variants (Overview)](#2-three-svm-variants-overview)
3. [L2 SVM: Full Derivation (Exercise 6a — 5 pts)](#3-l2-svm-full-derivation-exercise-6a--5-pts)
4. [How to Solve the L2 Dual (Exercise 6b — 5 pts)](#4-how-to-solve-the-l2-dual-exercise-6b--5-pts)
5. [Recovering w and b from KKT (Exercise 6c — 5 pts)](#5-recovering-w-and-b-from-kkt-exercise-6c--5-pts)
6. [The Kernel Trick (Exercise 5a — 5 pts)](#6-the-kernel-trick-exercise-5a--5-pts)
7. [Where the Kernel Trick Is Used (Exercise 5b — 5 pts)](#7-where-the-kernel-trick-is-used-exercise-5b--5-pts)
8. [Kernel Functions Reference](#8-kernel-functions-reference)
9. [Kernel SVMs](#9-kernel-svms)
10. [Exam Flashcard Summary](#10-exam-flashcard-summary)

---

## 1. Notation Reference

Before anything else, here is the notation the prof uses throughout. **The exam uses these exact symbols** — if you use different notation, you risk losing points.

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $X$ | Data matrix, columns are data points | $\mathbb{R}^{m \times n}$ |
| $y$ | Label vector, $y_j \in \{-1, +1\}$ | $\mathbb{R}^n$ |
| $w$ | Weight vector (normal to separating hyperplane) | $\mathbb{R}^m$ |
| $b$ | Bias (scalar) | $\mathbb{R}$ |
| $\rho$ | Margin (scalar, to be maximized) | $\mathbb{R}$ |
| $\xi$ | Slack variable vector, $\xi_j \geq 0$ | $\mathbb{R}^n$ |
| $z_j = y_j x_j$ | Label-scaled data point | $\mathbb{R}^m$ |
| $Z$ | Matrix of $z_j$ as columns, i.e. $Z = X \cdot \text{diag}(y)$ elementwise: each column $j$ of $Z$ is $y_j x_j$ | $\mathbb{R}^{m \times n}$ |
| $\mu$ | Lagrange multiplier vector | $\mathbb{R}^n$ |
| $C$ | Regularization parameter (tunes slack penalty) | $C > 0$ |
| $1$ | Vector of all ones | $\mathbb{R}^n$ |
| $I$ | Identity matrix | $\mathbb{R}^{n \times n}$ |
| $\Delta_{n-1}$ | Standard simplex: $\{\mu \geq 0, \; 1^\top \mu = 1\}$ | — |

> **Key notational detail:** $Z^\top Z$ is **not** the same as $X^\top X$. Since $z_j = y_j x_j$, we have $(Z^\top Z)_{ij} = z_i^\top z_j = y_i y_j \cdot x_i^\top x_j$. Equivalently, $Z^\top Z = (X^\top X) \odot (yy^\top)$, where $\odot$ is the element-wise (Hadamard) product.

---

## 2. Three SVM Variants (Overview)

The prof presents three SVM loss functions. They differ only in how they penalize slack:

| Variant | Primal Objective | Slack Penalty | Exam Relevance |
|---------|-----------------|---------------|----------------|
| **L1 SVM** | $\frac{1}{2}w^\top w + C \cdot 1^\top \xi$ | Linear in $\xi$ | Know it exists. Has ugly box constraint $0 \leq \mu \leq C1$ in dual. **Not asked on exam.** |
| **LSQ SVM** | $\frac{1}{2}(w^\top w + C \xi^\top \xi)$ | Squared $\xi$ | Solvable in closed form (linear system). Know it exists. **Not asked on exam.** |
| **L2 SVM** | $\frac{1}{2}(w^\top w + b^2 + C \xi^\top \xi) - \rho$ | Squared $\xi$, also penalizes $b^2$, maximizes margin $\rho$ | **THIS IS WHAT THE EXAM ASKS.** Full derivation below. |

> **Why the L2 SVM is "the real contender"** (prof's words): Its dual has a simple simplex constraint $\mu \in \Delta_{n-1}$ (no box constraint, no sum-to-zero). This makes it solvable by the Frank-Wolfe algorithm in ~13 lines of code.

---

## 3. L2 SVM: Full Derivation (Exercise 6a — 5 pts)

> **Exam question (verbatim from 1A/1B):** "Write down the (formal) Dual Problem of L2 SVM training."
>
> Even if you cannot reproduce the full derivation, **writing down just the final dual formula earns the 5 points**. But understanding the derivation lets you reconstruct it under pressure.

### Step 0: The Linear Binary Classifier

A linear binary classifier predicts:

$$f(x) = \text{sign}(w^\top x - b)$$

The SVM learns $w$ and $b$ by maximizing the margin $\rho$ between the two classes, while allowing some misclassification through slack variables $\xi_j \geq 0$.

---

### META-PATTERN Step 1: OBJECTIVE (Primal Problem)

The L2 SVM primal problem minimizes a combination of model complexity ($w^\top w + b^2$), misclassification penalty ($C \xi^\top \xi$), and negative margin ($-\rho$):

$$\boxed{\arg\min_{w, b, \rho, \xi} \quad \frac{1}{2}\left(w^\top w + b^2 + C \xi^\top \xi\right) - \rho}$$

$$\text{s.t.} \quad Z^\top w - by \geq \rho \cdot 1 - \xi$$

where the constraint says: each data point $j$ satisfies $y_j(w^\top x_j - b) \geq \rho - \xi_j$.

**Intuition:** We want to maximize the margin $\rho$ (hence $-\rho$ in the minimization), keep $w$ small (for generalization), keep $b$ small, and penalize misclassifications (large $\xi_j$) quadratically.

> **Partial credit note:** Writing this primal formulation already shows the examiner you understand the setup. Even if you can't derive the dual, this is worth writing down.

---

### META-PATTERN Step 2: CONSTRAINTS → Build the Lagrangian

We have $n$ inequality constraints (one per data point). We introduce Lagrange multipliers $\mu_j \geq 0$ (collected into vector $\mu \in \mathbb{R}^n$, $\mu \geq 0$) and form the Lagrangian:

$$L(w, b, \rho, \xi, \mu) = \frac{1}{2}\left(w^\top w + C \xi^\top \xi + b^2\right) - \rho - \sum_{j=1}^{n} \mu_j \left(y_j(w^\top x_j - b) - \rho + \xi_j\right)$$

Now we expand the sum. The term $\mu_j \cdot y_j(w^\top x_j - b)$ can be rewritten using $z_j = y_j x_j$:

$$\mu_j \cdot y_j \cdot w^\top x_j = \mu_j \cdot w^\top z_j$$

Summing over all $j$: $\sum_j \mu_j w^\top z_j = w^\top Z\mu$ (since $Z$ has $z_j$ as columns).

Similarly: $\sum_j \mu_j \cdot y_j \cdot b = b \cdot y^\top \mu$, and $\sum_j \mu_j \rho = \rho \cdot 1^\top \mu$, and $\sum_j \mu_j \xi_j = \xi^\top \mu$.

So the Lagrangian in compact matrix form is:

$$\boxed{L(w, b, \rho, \xi, \mu) = \frac{1}{2}\left(w^\top w + C\xi^\top \xi + b^2\right) - \rho - w^\top Z\mu + b \cdot y^\top \mu + \rho \cdot 1^\top \mu - \xi^\top \mu}$$

---

### META-PATTERN Step 3: DIFFERENTIATE — KKT Conditions (set partial derivatives to zero)

This is the heart of the derivation. We take the partial derivative of $L$ with respect to each primal variable and set it to zero.

**KKT w.r.t. $w$:**

We differentiate $L$ with respect to $w$. The terms involving $w$ are: $\frac{1}{2}w^\top w$ and $-w^\top Z\mu$.

Using the matrix calculus rule $\nabla_w(w^\top w) = 2w$, so $\nabla_w(\frac{1}{2}w^\top w) = w$.

And $\nabla_w(w^\top Z\mu) = Z\mu$ (since $Z\mu$ is a constant vector w.r.t. $w$).

$$\frac{\partial L}{\partial w} = w - Z\mu \stackrel{!}{=} 0 \quad \Longrightarrow \quad \boxed{w = Z\mu}$$

**KKT w.r.t. $b$:**

The terms involving $b$ are: $\frac{1}{2}b^2$ and $+b \cdot y^\top \mu$.

$$\frac{\partial L}{\partial b} = b + y^\top \mu \stackrel{!}{=} 0 \quad \Longrightarrow \quad \boxed{b = -y^\top \mu}$$

**KKT w.r.t. $\xi$:**

The terms involving $\xi$ are: $\frac{1}{2}C\xi^\top\xi$ and $-\xi^\top\mu$.

Using $\nabla_\xi(\xi^\top\xi) = 2\xi$, so $\nabla_\xi(\frac{1}{2}C\xi^\top\xi) = C\xi$.

$$\frac{\partial L}{\partial \xi} = C\xi - \mu \stackrel{!}{=} 0 \quad \Longrightarrow \quad \boxed{\xi = \frac{1}{C}\mu}$$

**KKT w.r.t. $\rho$:**

The terms involving $\rho$ are: $-\rho$ and $+\rho \cdot 1^\top\mu$.

$$\frac{\partial L}{\partial \rho} = -1 + 1^\top\mu \stackrel{!}{=} 0 \quad \Longrightarrow \quad \boxed{1^\top\mu = 1}$$

> **This is a critical result.** The constraint $1^\top \mu = 1$ together with $\mu \geq 0$ (from KKT complementarity) means $\mu$ lies on the **standard simplex** $\Delta_{n-1}$.

---

### META-PATTERN Step 4: SOLVE — Eliminate primal variables to get the dual

Now we substitute all four KKT results back into the Lagrangian to eliminate $w$, $b$, $\rho$, and $\xi$. This is pure algebra — follow each substitution carefully.

Starting from:

$$L = \frac{1}{2}\left(w^\top w + C\xi^\top\xi + b^2\right) - \rho - w^\top Z\mu + b \cdot y^\top\mu + \rho \cdot 1^\top\mu - \xi^\top\mu$$

**Substitute $w = Z\mu$:**

The term $\frac{1}{2}w^\top w$ becomes $\frac{1}{2}(Z\mu)^\top(Z\mu) = \frac{1}{2}\mu^\top Z^\top Z \mu$.

The term $-w^\top Z\mu$ becomes $-(Z\mu)^\top Z\mu = -\mu^\top Z^\top Z\mu$.

**Substitute $\xi = \frac{1}{C}\mu$:**

The term $\frac{1}{2}C\xi^\top\xi$ becomes $\frac{1}{2}C \cdot \frac{1}{C^2}\mu^\top\mu = \frac{1}{2C}\mu^\top\mu$.

The term $-\xi^\top\mu$ becomes $-\frac{1}{C}\mu^\top\mu$.

**Substitute $b = -y^\top\mu$:**

The term $\frac{1}{2}b^2$ becomes $\frac{1}{2}(y^\top\mu)^2 = \frac{1}{2}\mu^\top yy^\top\mu$.

The term $b \cdot y^\top\mu$ becomes $(-y^\top\mu)(y^\top\mu) = -(y^\top\mu)^2 = -\mu^\top yy^\top\mu$.

**Substitute $1^\top\mu = 1$:**

The term $-\rho + \rho \cdot 1^\top\mu = -\rho + \rho \cdot 1 = 0$. So $\rho$ drops out entirely.

Now we collect all terms:

$$L = \underbrace{\frac{1}{2}\mu^\top Z^\top Z\mu}_{\text{from } w^\top w} + \underbrace{\frac{1}{2C}\mu^\top\mu}_{\text{from } C\xi^\top\xi} + \underbrace{\frac{1}{2}\mu^\top yy^\top\mu}_{\text{from } b^2} \underbrace{- \mu^\top Z^\top Z\mu}_{\text{from } -w^\top Z\mu} \underbrace{- \frac{1}{C}\mu^\top\mu}_{\text{from } -\xi^\top\mu} \underbrace{- \mu^\top yy^\top\mu}_{\text{from } by^\top\mu}$$

Now combine like terms:

**$Z^\top Z$ terms:** $\frac{1}{2}\mu^\top Z^\top Z\mu - \mu^\top Z^\top Z\mu = -\frac{1}{2}\mu^\top Z^\top Z\mu$

**$\frac{1}{C}I$ terms:** $\frac{1}{2C}\mu^\top\mu - \frac{1}{C}\mu^\top\mu = -\frac{1}{2C}\mu^\top\mu$

**$yy^\top$ terms:** $\frac{1}{2}\mu^\top yy^\top\mu - \mu^\top yy^\top\mu = -\frac{1}{2}\mu^\top yy^\top\mu$

So the Lagrangian reduces to:

$$L = -\frac{1}{2}\mu^\top Z^\top Z\mu - \frac{1}{2C}\mu^\top\mu - \frac{1}{2}\mu^\top yy^\top\mu$$

Factor out $-\frac{1}{2}\mu^\top(\cdots)\mu$:

$$L = -\frac{1}{2}\mu^\top\left(Z^\top Z + yy^\top + \frac{1}{C}I\right)\mu$$

---

### The Dual Problem of L2 SVM Training

> **THIS IS THE FORMULA THE EXAM ASKS YOU TO WRITE DOWN VERBATIM (Exercise 6a, 5 pts).**

$$\boxed{\arg\min_{\mu} \quad \mu^\top \left[Z^\top Z + yy^\top + \frac{1}{C}I\right] \mu \qquad \text{s.t.} \quad 1^\top\mu = 1, \quad \mu \geq 0}$$

Or equivalently, in the prof's most compact notation:

$$\boxed{\arg\min_{\mu \in \Delta_{n-1}} \quad \mu^\top \left[Z^\top Z + yy^\top + \frac{1}{C}I\right] \mu}$$

where $\Delta_{n-1} = \{\mu \in \mathbb{R}^n : \mu \geq 0, \; 1^\top\mu = 1\}$ is the standard simplex.

> **Common mistake:** Students forget the $yy^\top$ term or the $\frac{1}{C}I$ term. The matrix inside the brackets has **three** additive components. Memorize: **$Z^\top Z$ + $yy^\top$ + $\frac{1}{C}I$**.

> **Note on the factor of $\frac{1}{2}$:** The prof's slides sometimes write $\mu^\top[\cdots]\mu$ without the $\frac{1}{2}$ (slide 33-34). Since we're doing $\arg\min$, the factor of $\frac{1}{2}$ doesn't change the minimizer, so both forms are correct. Write whichever you remember.

---

## 4. How to Solve the L2 Dual (Exercise 6b — 5 pts)

> **Exam question (verbatim):** "How would you solve this problem?"

The dual is a **quadratic program on the simplex**: minimize $\mu^\top M \mu$ subject to $\mu \in \Delta_{n-1}$, where $M = Z^\top Z + yy^\top + \frac{1}{C}I$.

The answer is: **Use the Frank-Wolfe algorithm.**

### What to write on the exam:

The Frank-Wolfe algorithm solves optimization problems over convex sets (here, the simplex). At each iteration:

1. **Linearize** the objective at the current point: compute the gradient $\nabla f(\mu) = 2M\mu$
2. **Minimize the linear approximation** over the simplex. For the simplex, this has a closed-form solution: the minimum is at the vertex $e_s$ (the $s$-th standard basis vector) where $s = \arg\min_j (\nabla f(\mu))_j$
3. **Step toward that vertex**: $\mu^{(t+1)} = \mu^{(t)} + \beta_t (e_s - \mu^{(t)})$, where $\beta_t = \frac{2}{t+2}$

> **The key insight is:** On the simplex, the Frank-Wolfe linear subproblem reduces to finding the coordinate with the smallest gradient component. This is just an `argmin` — no QP solver needed.

> **Partial credit note:** Even if you don't remember the update rule details, writing "Frank-Wolfe algorithm on the simplex" and saying "iterative method that linearizes the objective and steps toward simplex vertices" gets you most of the 5 points.

---

## 5. Recovering w and b from KKT (Exercise 6c — 5 pts)

> **Exam question (verbatim):** "Using the Karush-Kuhn-Tucker conditions: Once you have solved the L2 training problem, how can $w$ and $b$ be computed?"

These come directly from KKT conditions derived in Step 3 above:

$$\boxed{w = Z\mu = \sum_{j=1}^{n} \mu_j z_j = \sum_{j=1}^{n} \mu_j y_j x_j}$$

$$\boxed{b = -y^\top\mu = -\sum_{j=1}^{n} \mu_j y_j}$$

**Expanding $w = Z\mu$:** Since $Z$ has columns $z_j = y_j x_j$, multiplying by $\mu$ gives a weighted sum of label-scaled data points. Only the **support vectors** (those $x_j$ where $\mu_j > 0$) contribute.

> **Students lose points here** by writing $w = X\mu$ instead of $w = Z\mu$. Remember: $Z$, not $X$. Each column of $Z$ is $y_j x_j$, not just $x_j$.

> **Connection:** This formula $w = \sum_j \mu_j y_j x_j$ is what makes kernelization possible — $w$ is expressed purely as a weighted combination of data points. During prediction, the decision function becomes $f(x) = \text{sign}(w^\top x - b) = \text{sign}\left(\sum_j \mu_j y_j (x_j^\top x) + y^\top\mu\right)$, and every data point appears inside an inner product $x_j^\top x$. This is the entry point for the kernel trick (Exercise 5).

---

## 6. The Kernel Trick (Exercise 5a — 5 pts)

> **Exam question (verbatim from 1A/1B):** "What is the kernel trick?"

### Intuition

Linear classifiers like SVMs can only draw straight decision boundaries. Real data often isn't linearly separable. The kernel trick lets us use a linear algorithm to solve a **non-linear** problem, without ever explicitly computing the high-dimensional feature transformation.

### The Definition (Two Steps)

A **Mercer kernel** is a positive semidefinite function $k : \mathbb{R}^m \times \mathbb{R}^m \to \mathbb{R}$ for which there exists a feature map $\phi : \mathbb{R}^m \to \mathbb{R}^M$ (where $M$ can be very large or even $\infty$) such that:

$$k(x, y) = \phi(x)^\top \phi(y)$$

The kernel trick is a two-step procedure:

$$\boxed{\textbf{Step 1: Rewrite the algorithm so that data points only appear inside inner products } x_i^\top x_j}$$

$$\boxed{\textbf{Step 2: Replace every inner product } x_i^\top x_j \textbf{ with a kernel evaluation } k(x_i, x_j)}$$

**Why this works:** By Mercer's theorem, the kernel evaluation $k(x_i, x_j)$ implicitly computes the inner product $\phi(x_i)^\top \phi(x_j)$ in a high-dimensional feature space $\mathbb{R}^M$ — **without ever computing $\phi(x_i)$ or $\phi(x_j)$ explicitly**. This avoids the computational cost of working in $\mathbb{R}^M$ (which can be infinite-dimensional for the Gaussian kernel).

### What to write on the exam (5 pts):

> "The kernel trick has two steps: (1) rewrite the algorithm so that input data only appears in the form of inner products with other data, and (2) replace each inner product $x_i^\top x_j$ by a kernel evaluation $k(x_i, x_j)$. A Mercer kernel is a positive semidefinite function $k(x,y) = \phi(x)^\top\phi(y)$ that implicitly computes inner products in a high-dimensional feature space without explicitly computing the feature map $\phi$. This allows linear methods to handle non-linear problems."

---

## 7. Where the Kernel Trick Is Used (Exercise 5b — 5 pts)

> **Exam question (verbatim):** "Where and how is the Kernel trick used in Machine Learning?"

### Answer pattern (give 3-4 examples with one line each):

1. **Kernel SVMs** — Replace the Gram matrix $X^\top X$ (entries $x_i^\top x_j$) with a kernel matrix $K$ (entries $k(x_i, x_j)$). This lets SVMs learn non-linear decision boundaries. For the L2 SVM dual, $Z^\top Z$ becomes $K \odot yy^\top$.

2. **Kernel Least Squares Regression** — The closed-form regression solution involves $X^\top X$; replacing it with $K$ gives kernelized ridge regression: $\hat{f}(x) = k(x)^\top(K + \lambda I)^{-1}y$.

3. **Kernel PCA** — Standard PCA uses the covariance matrix $X^\top X$; kernel PCA replaces it with a kernel matrix $K$ to find non-linear principal components.

4. **Gaussian Processes** — The GP covariance matrix $C(\theta)$ is a kernel matrix. GP regression is equivalent to kernelized regression.

> **Connection to Exercises:** This answer bridges Exercise 5 (kernel trick), Exercise 6 (kernel SVM), and Exercise 7 (GPs use kernels). Mentioning this chain shows the examiner you understand the course structure.

### How it works concretely in SVMs:

In training, the Gram matrix $X^\top X$ (whose entries are inner products $x_i^\top x_j$) appears in the dual. We replace it:

$$X^\top X \quad \longrightarrow \quad K, \quad \text{where } K_{ij} = k(x_i, x_j)$$

For the L2 SVM specifically:

$$\underbrace{Z^\top Z + yy^\top + \frac{1}{C}I}_{\text{linear}} \quad \longrightarrow \quad \underbrace{K \odot yy^\top + yy^\top + \frac{1}{C}I}_{\text{kernelized}}$$

In application (prediction), the vector $x^\top X$ (inner products of test point with all training points) becomes $k(x)^\top$ where $k(x)_j = k(x, x_j)$.

---

## 8. Kernel Functions Reference

These are the three most important kernels. Know the formula and what $M$ (feature space dimension) is:

| Kernel | Formula | Feature Space Dim $M$ | Notes |
|--------|---------|----------------------|-------|
| **Linear** | $k(x,y) = x^\top y$ | $m$ (same as input) | Trivial kernel, $\phi(x) = x$ |
| **Polynomial** | $k(x,y) = (b + x^\top y)^d$ | $\binom{m+d}{d}$ | $b > 0$, $d \in \mathbb{N}$. Inhomogeneous linear is the $d=1$ case. |
| **Gaussian (RBF)** | $k(x,y) = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$ | $\infty$ | The most powerful; an infinite sum over polynomial kernels |

**How to build valid kernels (kernel engineering):** Given valid kernels $k_1, k_2$ and constants $c > 0$, $b > 0$, function $g$:

- $c \cdot k_1(x,y)$ is valid
- $k_1(x,y) + k_2(x,y)$ is valid
- $k_1(x,y)^d$ is valid (for $d \in \mathbb{N}$)
- $g(x) \cdot k_1(x,y) \cdot g(y)$ is valid

> **Exam relevance:** If asked "name a kernel" or "give an example," use the Gaussian kernel — it's the most interesting (infinite-dimensional feature space) and the one used in exercises.

---

## 9. Kernel SVMs

### Kernelized L2 SVM Training

Replace the matrix in the dual:

$$M = Z^\top Z + yy^\top + \frac{1}{C}I \quad \longrightarrow \quad M = K \odot yy^\top + yy^\top + \frac{1}{C}I$$

where $K_{ij} = k(x_i, x_j)$.

> **Why $K \odot yy^\top$?** Recall $Z^\top Z = (X^\top X) \odot (yy^\top)$. When kernelizing, $X^\top X \to K$, so $Z^\top Z \to K \odot yy^\top$.

### Kernelized L2 SVM Prediction

The decision function becomes:

$$f(x) = \text{sign}\left(\sum_{x_s \in S} \mu_s y_s \, k(x, x_s) + y^\top\mu\right)$$

where $S$ is the set of **support vectors** (training points with $\mu_j > 0$).

> **Efficiency:** Only support vectors contribute to the sum. Non-support vectors have $\mu_j = 0$, so they can be dropped.

---

## 10. Exam Flashcard Summary

### Exercise 6a: "Write down the L2 SVM dual" (5 pts)

$$\arg\min_{\mu \in \Delta_{n-1}} \mu^\top \left[Z^\top Z + yy^\top + \frac{1}{C}I\right]\mu$$

where $\Delta_{n-1} = \{\mu \geq 0, \; 1^\top\mu = 1\}$, $Z$ has columns $z_j = y_j x_j$.

### Exercise 6b: "How would you solve it?" (5 pts)

**Frank-Wolfe algorithm** on the simplex. Iterative: compute gradient, step toward the simplex vertex with the smallest gradient component.

### Exercise 6c: "How to recover $w$ and $b$?" (5 pts)

$$w = Z\mu \qquad b = -y^\top\mu$$

### Exercise 5a: "What is the kernel trick?" (5 pts)

Two steps: (1) rewrite algorithm so data appears only in inner products $x_i^\top x_j$; (2) replace inner products with kernel evaluations $k(x_i, x_j)$, where $k(x,y) = \phi(x)^\top\phi(y)$ for some feature map $\phi$.

### Exercise 5b: "Where is it used?" (5 pts)

Kernel SVMs, kernel ridge regression, kernel PCA, Gaussian processes. Everywhere an inner product/Gram matrix $X^\top X$ appears, it can be replaced by a kernel matrix $K$.

---

## What This Lecture Does NOT Cover (Skip These)

- **L1 SVM details** — Not asked on exam. Has ugly box constraint. One sentence: "L1 SVM uses linear slack penalty, leading to a box constraint $0 \leq \mu \leq C1$ in the dual."
- **LSQ SVM derivation** — Not asked directly. Know it exists and has a closed-form linear system solution.
- **All Python code** — 0 exam points. The `trainL2SVM` numpy code is for exercise sheets only.
- **Proofs of kernel validity** (Mercer's theorem proof, spectral decomposition of kernels) — not tested. Just know the statement.
- **Boosting / neural networks as alternative strategies** — mentioned on slides 41-43 for context only. Not tested.

---

## Connections to Other Exercises

| This Lecture's Content | Connects To | How |
|----------------------|-------------|-----|
| L2 SVM dual derivation | **Ex 4** (optimization) | Same meta-pattern: objective → Lagrangian → differentiate → solve |
| KKT conditions | **Ex 4** (constrained optimization from lect-09) | The Lagrangian/KKT technique is the same for both |
| Kernel trick | **Ex 7** (Gaussian processes) | GP covariance = kernel matrix. GP regression = kernel regression |
| $w = Z\mu$ (weight as weighted sum of data) | **Ex 3** (Bayesian, MAP) | MAP with Gaussian prior gives $w = (X X^\top + \lambda I)^{-1}Xy$, also expressible via data |
| Frank-Wolfe on simplex | **Ex 7** (non-convex optimization) | Ex 7 uses gradient descent; Frank-Wolfe is a related iterative method |
| Regularization parameter $C$ | **Ex 3** (MAP, ridge regression) | $C$ controls regularization strength, just like $\lambda$ in ridge regression. $\frac{1}{C}I$ is the regularization term. |
