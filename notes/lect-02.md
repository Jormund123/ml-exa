# Lecture 02 — Least Squares Regression

**Source:** `slides/lect-02.txt` (79 slides)
**Exam mapping:** Foundations for **Exercise 1b** (loss function), **Exercise 4** (optimization/meta-pattern), and general background for all later exercises
**Priority:** MEDIUM — teaches the meta-pattern skeleton and notation used everywhere

---

## Why This Lecture Matters for the Exam

This lecture doesn't directly map to a single high-point exercise, BUT it introduces:

1. **The least squares loss** — the archetype of all loss functions (Ex 1b)
2. **The "differentiate and set to zero" method** — the meta-pattern you'll use in Ex 4, 6, 7
3. **Feature maps and feature matrices $\Phi$** — notation used in SVM (Ex 6) and kernel methods (Ex 5, 7)
4. **The closed-form solution $\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y$** — reappears in kernel regression / GPs (Ex 7)

---

## Exam-Relevant Formulas

### 1. The Least Squares Loss (→ Ex 1b)

Given data $D = \{(x_j, y_j)\}_{j=1}^n$ and a model $f(x|w) = w^\top \phi(x)$:

**Residual / Error:**
$$\epsilon_j = f_\theta(x_j) - y_j$$

**Mean Squared Error:**
$$E(w) = \frac{1}{n}\sum_{j=1}^{n} (f_\theta(x_j) - y_j)^2$$

**Least Squares Loss (equivalent, just drops the $1/n$):**
$$L(w) = \sum_{j=1}^{n} (\phi_j^\top w - y_j)^2 = \|\Phi^\top w - y\|^2$$

> **Notation clarification — why $\phi_j^\top w$ instead of $w^\top \phi_j$?**
>
> The model is $f(x|w) = w^\top \phi(x)$. Since a dot product commutes for real vectors, $w^\top \phi_j = \phi_j^\top w$ — both equal the scalar $\sum_{i=1}^{M} w_i \phi_i(x_j)$.
>
> The prof writes $\phi_j^\top w$ because it makes the **matrix form** work: row $j$ of $\Phi^\top$ is $\phi_j^\top$, so the matrix-vector product $\Phi^\top w$ stacks all predictions into one vector:
>
> $$\Phi^\top w = \begin{bmatrix} \phi_1^\top w \\ \vdots \\ \phi_n^\top w \end{bmatrix} = \begin{bmatrix} f(x_1|w) \\ \vdots \\ f(x_n|w) \end{bmatrix}$$
>
> This is just a notational convenience to go from per-point scalars to the compact vector form $\|\Phi^\top w - y\|^2$.

> **Exam tip for Ex 1b:** "A loss function $L(\theta | f, D)$ measures how well model $f$ fits data $D$. The smaller the loss, the better the fit. Example: the least squares loss $L(w) = \sum_j (f(x_j) - y_j)^2$ penalizes squared prediction errors."

---

### 2. Feature Maps and Feature Matrices (→ notation for Ex 5, 6, 7)

**Feature map** $\phi: \mathbb{R}^m \to \mathbb{R}^M$ transforms input data into a (possibly higher-dimensional) space.

Examples from this lecture:

| Regression Type         | Feature Map $\phi(x)$                                         | Dimension $M$ |
| ----------------------- | ------------------------------------------------------------- | ------------- |
| Linear                  | $\phi(x) = [1, x]^\top$                                       | 2             |
| Polynomial (degree $d$) | $\phi(x) = [1, x, x^2, \ldots, x^d]^\top$                     | $d+1$         |
| RBF                     | $\phi_j(x) = [\varphi(x, x_1), \ldots, \varphi(x, x_n)]^\top$ | $n$           |

**Feature matrix:**
$$\Phi = [\phi_1 \; \phi_2 \; \cdots \; \phi_n] \in \mathbb{R}^{M \times n}$$

where $\phi_j = \phi(x_j)$ are column vectors.

> **This notation is the prof's convention and it's used throughout the entire course.** When you see $\Phi$ in Ex 5, 6, 7 — this is what it means.

> **Connection to Ex 5 (kernel trick):** The kernel trick says: instead of explicitly computing $\phi(x)$ and forming $\Phi$, just compute inner products $\phi(x_i)^\top \phi(x_j) = k(x_i, x_j)$ directly.

---

### 3. The Least Squares Solution — META-PATTERN IN ACTION (→ Ex 4)

**Problem:** We want to find the weight vector $w$ that minimizes the squared error between our predictions $\Phi^\top w$ and the targets $y$:

$$\hat{w} = \arg\min_w \|\Phi^\top w - y\|^2$$

**Meta-pattern applied:**

**Step 1 — OBJECTIVE:** First, expand the squared norm into a form we can differentiate. Recall that $\|a\|^2 = a^\top a$, so we write out the product and multiply through:

$$L(w) = \|\Phi^\top w - y\|^2 = (\Phi^\top w - y)^\top(\Phi^\top w - y)$$

Let's expand this step by step. Let $a = \Phi^\top w$ and $b = y$, so we need $(a - b)^\top(a - b)$. This works like $(a-b)^2 = a^2 - 2ab + b^2$, but with transposes:

$$(a - b)^\top(a - b) = a^\top a - a^\top b - b^\top a + b^\top b$$

Now substitute back $a = \Phi^\top w$ and $b = y$, and simplify each of the four terms:

**Term $a^\top a$:** $(\Phi^\top w)^\top (\Phi^\top w)$. Use the transpose rule $(AB)^\top = B^\top A^\top$, so $(\Phi^\top w)^\top = w^\top (\Phi^\top)^\top = w^\top \Phi$. Therefore:

$$(\Phi^\top w)^\top (\Phi^\top w) = w^\top \Phi \cdot \Phi^\top w = w^\top \Phi\Phi^\top w$$

**Term $-a^\top b$:** $-(\Phi^\top w)^\top y = -w^\top \Phi y$ (same transpose rule as above).

**Term $-b^\top a$:** $-y^\top \Phi^\top w$. This is a scalar, and the transpose of a scalar is itself, so $y^\top \Phi^\top w = (w^\top \Phi y)^\top = w^\top \Phi y$. Therefore this term also equals $-w^\top \Phi y$.

**Combining the two cross-terms:** $-w^\top \Phi y - w^\top \Phi y = -2w^\top \Phi y$

**Term $b^\top b$:** $y^\top y$ (unchanged).

**Final result:**

$$L(w) = w^\top \Phi\Phi^\top w - 2w^\top \Phi y + y^\top y$$

**Step 2 — CONSTRAINTS:** None — this is unconstrained optimization, so we skip the Lagrangian and go straight to differentiating.

**Step 3 — DIFFERENTIATE and set to zero:** We differentiate $L(w) = w^\top \Phi\Phi^\top w - 2w^\top \Phi y + y^\top y$ term by term with respect to $w$:

**Term 1:** $w^\top \Phi\Phi^\top w$ — this has the form $w^\top A w$ where $A = \Phi\Phi^\top$. Note that $A$ is symmetric (because $(B B^\top)^\top = B B^\top$ for any matrix $B$). The matrix calculus rule is: $\nabla_w(w^\top A w) = 2Aw$ when $A$ is symmetric. So:

$$\nabla_w(w^\top \Phi\Phi^\top w) = 2\Phi\Phi^\top w$$

**Term 2:** $-2w^\top \Phi y$ — this has the form $w^\top b$ where $b = \Phi y$ is just a constant vector (no $w$ in it). The rule is: $\nabla_w(w^\top b) = b$. So:

$$\nabla_w(-2w^\top \Phi y) = -2\Phi y$$

**Term 3:** $y^\top y$ — this is a constant (no $w$ at all), so its gradient is zero.

**Combining all three terms:**

$$\nabla_w L = 2\Phi\Phi^\top w - 2\Phi y + 0 \stackrel{!}{=} 0$$

**Step 4 — SOLVE:** Divide both sides by 2. Then, assuming $\Phi\Phi^\top$ is invertible, multiply both sides on the left by $(\Phi\Phi^\top)^{-1}$:

$$\Phi\Phi^\top w = \Phi y$$
$$\boxed{\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y}$$

> **This is THE template for Ex 4.** Exercise 4 asks you to minimize $\sum \|x_j - x\|^2$. Same skeleton: write objective → differentiate → set to zero → solve. Here is the full derivation:

**Ex 4 — Finding the point that minimizes squared distances (sample mean)**

**Problem:** Given data points $x_1, x_2, \ldots, x_n \in \mathbb{R}^m$, find the point $\hat{x}$ that minimizes the total squared distance to all data points:

$$\hat{x} = \arg\min_x \sum_{j=1}^{n} \|x_j - x\|^2$$

**Step 1 — OBJECTIVE:** Expand the squared norm using $\|a\|^2 = a^\top a$:

$$L(x) = \sum_{j=1}^{n} \|x_j - x\|^2 = \sum_{j=1}^{n} (x_j - x)^\top(x_j - x)$$

Expand each term (same $(a-b)^2$ trick as before):

$$= \sum_{j=1}^{n} \left( x_j^\top x_j - 2x_j^\top x + x^\top x \right)$$

**Step 2 — CONSTRAINTS:** None (unconstrained).

**Step 3 — DIFFERENTIATE and set to zero:** Differentiate with respect to $x$ term by term:

- $x_j^\top x_j$ → constant (no $x$), gradient is $0$
- $-2x_j^\top x$ → has the form $b^\top x$, so gradient is $-2x_j$
- $x^\top x$ → gradient is $2x$

Sum over all $j$:

$$\nabla_x L = \sum_{j=1}^{n} (0 - 2x_j + 2x)$$

Split the sum — the $-2x_j$ part depends on $j$, but $2x$ does not, so summing $2x$ a total of $n$ times just gives $n \cdot 2x$:

$$= -2\sum_{j=1}^{n} x_j + \underbrace{\sum_{j=1}^{n} 2x}_{n \text{ copies of } 2x} = -2\sum_{j=1}^{n} x_j + 2nx \stackrel{!}{=} 0$$

**Step 4 — SOLVE:**

$$2nx = 2\sum_{j=1}^{n} x_j$$

$$\boxed{\hat{x} = \frac{1}{n}\sum_{j=1}^{n} x_j}$$

> This is just the **sample mean** — the arithmetic average of all data points minimizes the sum of squared distances. Same meta-pattern, simpler algebra.

> **This also previews Ex 6 (SVM):** The SVM dual is derived with the same pattern but WITH constraints, so you build a Lagrangian first before differentiating.

---

### 4. The Expanded Derivation (step by step as you'd write on the exam)

$$L = \|\Phi^\top w - y\|^2$$
$$= (\Phi^\top w - y)^\top (\Phi^\top w - y)$$
$$= w^\top \Phi \Phi^\top w - 2w^\top \Phi y + y^\top y$$

Differentiate w.r.t. $w$:
$$\frac{dL}{dw} = 2\Phi\Phi^\top w - 2\Phi y$$

Set to zero:
$$2\Phi\Phi^\top w - 2\Phi y = 0$$
$$\Phi\Phi^\top w = \Phi y$$
$$w = (\Phi\Phi^\top)^{-1} \Phi y$$

> **Partial credit note:** Even writing $L = \|\Phi^\top w - y\|^2$ and then "take derivative and set to zero" earns points. The setup IS the hard part.

---

### 5. RBF Regression (→ background for Ex 5, 7)

RBF stands for **Radial Basis Function**. The idea is: instead of using polynomial features like $[1, x, x^2, \ldots]$, we place a "bump" (a Gaussian bell curve) centered at each data point $x_j$, and our model is a weighted sum of these bumps:

$$f(x|w) = \sum_{j=1}^{n} \varphi(x, x_j) \cdot w_j$$

where $\varphi(x, x_j) = e^{-\frac{1}{\beta}\|x - x_j\|^2}$ is a Gaussian RBF centered at data point $x_j$. It outputs a value close to 1 when $x$ is near $x_j$, and decays toward 0 as $x$ moves away. The parameter $\beta$ controls how wide each bump is.

Now, to solve for the weights $w$, we use our least squares solution $\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y$. But something nice happens here: the feature matrix $\Phi$ has entries $\Phi_{ij} = \varphi(x_i, x_j)$. Since $\varphi(x_i, x_j) = \varphi(x_j, x_i)$ (the distance $\|x_i - x_j\|$ is symmetric), the matrix $\Phi$ is **square** ($n \times n$, because we have one basis function per data point) **and symmetric** ($\Phi = \Phi^\top$).

Because $\Phi$ is square and symmetric, $\Phi^\top = \Phi$, so $\Phi\Phi^\top = \Phi \cdot \Phi = \Phi^2$. The solution simplifies:

$$\hat{w} = (\Phi^2)^{-1}\Phi y = \Phi^{-2}\Phi y = \Phi^{-1} y$$

$$\boxed{\hat{w} = \Phi^{-1} y}$$

> In other words: with RBF features, finding the weights reduces to just solving a linear system $\Phi w = y$. No need for the full $(\Phi\Phi^\top)^{-1}\Phi$ formula.

> **Connection to Ex 7 (GPs):** This RBF matrix $\Phi_{ij} = \varphi(x_i, x_j)$ is essentially a **kernel matrix**. In GP regression, the covariance matrix $C(\Theta)$ plays the same role. Kernel regression and GP regression are fundamentally the same thing.

> **Connection to Ex 5 (kernel trick):** The RBF $\varphi(x_i, x_j)$ is a kernel function. Wherever you see inner products $x_i^\top x_j$, you can replace them with $k(x_i, x_j)$ — this is the kernel trick.

---

### 6. Notation Convention (used throughout the course)

| Symbol                             | Meaning                                                |
| ---------------------------------- | ------------------------------------------------------ |
| $n$                                | number of data points                                  |
| $m$                                | dimension of input data vectors $x_j \in \mathbb{R}^m$ |
| $M$                                | dimension of feature vectors $\phi_j \in \mathbb{R}^M$ |
| $\phi(x)$ or $\phi_x$              | feature map applied to input $x$                       |
| $\phi_j$                           | shorthand for $\phi(x_j)$                              |
| $\Phi \in \mathbb{R}^{M \times n}$ | feature matrix (columns = feature vectors)             |
| $\Phi^\top$ or $\Psi$              | design matrix ($n \times M$, rows = feature vectors)   |
| $y \in \mathbb{R}^n$               | target vector                                          |
| $w \in \mathbb{R}^M$               | parameter vector                                       |
| $\theta$                           | abstract parameter set                                 |

---

## What to Skip

- **Historical remarks about Gauss** (slide 3) — 0 exam points
- **Height/weight example details** — just a didactic vehicle
- **Vandermonde matrix** (slide 45) — unlikely exam topic
- **Numerical warnings about invertibility** (slide 67) — practical issue, not tested
- **Neural network analogy** (slide 79) — cute but not tested
- **QR/SVD for solving least squares** (slide 67) — implementation detail, skip

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 02:

| #   | Formula                                                | Maps to                                                                     |
| --- | ------------------------------------------------------ | --------------------------------------------------------------------------- |
| 1   | $L(w) = \|\Phi^\top w - y\|^2$                         | **Ex 1b** (loss function example)                                           |
| 2   | $\nabla_w L = 2\Phi\Phi^\top w - 2\Phi y = 0$          | **Ex 4** (differentiate and set to zero — the meta-pattern)                 |
| 3   | $\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y$                 | **Ex 4** (closed-form solution), **Ex 7** (kernel regression/GP background) |
| 4   | Feature map: $\phi(x) = [1, x, x^2, \ldots, x^d]^\top$ | **Ex 5** (kernel trick background — feature maps)                           |
| 5   | RBF: $\varphi(x, c) = e^{-\frac{1}{\beta}\|x-c\|^2}$   | **Ex 5, 7** (kernel function, GP covariance)                                |

### Verdict:

**This lecture teaches the METHOD you'll use everywhere.** The specific formulas (least squares solution) are less likely to be asked directly, but the **meta-pattern** (objective → differentiate → set to zero → solve) IS the exam.

**Drill this:** Given any objective function, can you differentiate it, set to zero, and solve? If yes, you can handle Ex 4 (10 pts), contribute to Ex 6 (15 pts), and contribute to Ex 7 (15 pts). That's 40 points riding on this one skill.

**Then move on to lect-05** — that's where the actual high-point formulas (likelihood, MLE, MAP, Bayes) live.
