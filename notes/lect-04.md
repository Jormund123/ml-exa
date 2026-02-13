# Lecture 04 — Binary Classification, Logistic Regression, and Autodiff

**Source:** `slides/lect-04.txt` (76 slides)
**Exam mapping:** Minor support for **Exercise 1b** (loss function example). Conceptual background for gradient-based training (Ex 6b, 7b). Chain rule / autodiff is not tested.
**Priority:** LOW — no formulas from this lecture appear directly on past exams

---

## Exam-Relevant Extractions

### 1. The Logistic Loss — Another Loss Function Example (→ Exercise 1b)

Binary classifier model:

$$f(x) = \text{sign}(x^\top w - b)$$

Since $\text{sign}$ is discontinuous, replace with the smooth surrogate $\tanh$:

$$f(x) = \tanh\left(\beta(x^\top w - b)\right)$$

The **logistic loss** (what we minimize to train the classifier):

$$L(w, b, \beta | D) = \frac{1}{n}\sum_{j=1}^{n} \log\left(1 + e^{-2\beta y_j(x_j^\top w - b)}\right)$$

**Why this matters for Ex 1b:** If the exam asks "give an example of a loss function," the squared loss from lect-02 is the safest answer. But the logistic loss is a second example showing that different loss functions → different classifiers. The key insight from this lecture:

> Different loss functions produce different classifiers (LDA, perceptron, logistic regression, SVM) even for the same task. The choice of loss is a design decision.

**Connection to Ex 6:** The SVM uses the hinge loss, not the logistic loss. But both are loss functions for binary classification. The logistic loss is always convex → unique global minimizer.

---

### 2. Gradient Descent Applied to Logistic Regression (→ background for Ex 6b, 7b)

The prof trains the logistic regression classifier via gradient descent:

$$w_{k+1} = w_k - \eta \cdot \frac{1}{n}\sum_{j=1}^{n} \frac{\partial L_j}{\partial w}$$

The partial derivatives are computed via the **chain rule**:

$$\frac{\partial L_j}{\partial w} = \frac{\partial L_j}{\partial g_j} \cdot \frac{\partial g_j}{\partial h_j} \cdot \frac{\partial h_j}{\partial w} = \frac{1}{g_j} \cdot e^{h_j} \cdot (-2\beta y_j x_j)$$

where $h_j = -2\beta y_j(x_j^\top w - b)$ and $g_j = 1 + e^{h_j}$.

> **Exam relevance:** You will NOT be asked to derive these partials. But the concept — "compute gradient, update parameters" — is what you'll reference in Ex 6b ("How would you solve the SVM dual?" → Frank-Wolfe) and Ex 7b ("How would you solve GP optimization?" → gradient ascent + random restarts).

---

### 3. Autodiff / Computation Graphs (→ NOT tested)

The second half of the lecture covers:
- Computation graphs (DAGs of operations)
- Forward pass (evaluate function)
- Backward pass (propagate gradients via chain rule)
- Adjoint notation: $\bar{v}_k = \frac{\partial v_K}{\partial v_k}$
- JAX implementation

> **Exam relevance: ZERO.** Autodiff is a practical tool, not a pen-and-paper exam topic. The prof says "work with autodiff libraries wherever possible" — this is advice for coding, not for the exam. **Skip entirely.**

---

### 4. Key Concepts (one-liners, know they exist)

| Concept | What to Know | Exam Use |
|---------|-------------|----------|
| Linear binary classifier | $f(x) = \text{sign}(x^\top w - b)$; separating hyperplane | Background for Ex 6 (SVM is a specific type) |
| Bipolar labels $y_j \in \{-1, +1\}$ | The prof's convention throughout the course | Notation for Ex 6 (SVM uses this) |
| Logistic function $\sigma(\gamma x) = \frac{1}{1+e^{-\gamma x}}$ | Sigmoid, maps $\mathbb{R} \to (0,1)$ | Not directly tested |
| Surrogate functions | Replace discontinuous $\text{sign}$ with smooth $\tanh$ | Not tested, but the principle (relax hard problems) recurs |
| Regularized loss $L = L_l + L_r$ where $L_r = \frac{1}{2}w^\top w$ | Regularization term added to loss | Connection to Ex 3: regularization = MAP with Gaussian prior |

---

## Connections to Exam Exercises

| Exercise | What This Lecture Contributes |
|----------|------------------------------|
| **Ex 1b** | Another loss function example (logistic loss). Use squared loss as primary answer, mention logistic as second example if needed. |
| **Ex 3** | $L_r = \frac{1}{2}w^\top w$ is L2 regularization = MAP with Gaussian prior. This connection is from lect-06/07 but foreshadowed here. |
| **Ex 6** | SVM is another linear binary classifier. Same setup ($y_j \in \{-1,+1\}$, separating hyperplane $x^\top w - b$) but different loss (hinge loss, not logistic). |

---

## What to Skip

- **ALL Python code** (numpy, JAX implementations) — 0 exam points
- **Autodiff theory** (computation graphs, forward/backward pass, adjoint notation) — 0 exam points
- **Symbolic differentiation with sympy** — 0 exam points
- **Detailed logistic loss gradient derivation** — not asked on the exam
- **Historical remarks about deep learning revolution** — 0 exam points

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 04:

| # | Formula/Concept | Maps to |
|---|----------------|---------|
| 1 | $f(x) = \text{sign}(x^\top w - b)$ (linear binary classifier) | **Ex 6** (SVM background — same model family) |
| 2 | Logistic loss: $L = \frac{1}{n}\sum_j \log(1 + e^{-2\beta y_j(x_j^\top w - b)})$ | **Ex 1b** (secondary loss function example) |
| 3 | Bipolar labels $y_j \in \{-1, +1\}$ | **Ex 6** (SVM notation uses this) |
| 4 | Regularization $L_r = \frac{1}{2}w^\top w$ | **Ex 3** (connection: regularization = MAP) |

### Verdict:

**Low priority — know it exists, don't memorize.** No formula from this lecture is directly asked on past exams. The binary classifier setup and bipolar labels are useful notation for understanding Ex 6 (SVM), but you'll learn those properly in lect-10/11. The logistic loss derivation is a good worked example of gradient descent + chain rule, but it's not what the exam tests.

**Move on to lect-05 — that's where 30 points of exam content live.**
