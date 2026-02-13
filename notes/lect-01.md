# Lecture 01 — A First Glimpse at Machine Learning

**Source:** `slides/lect-01.txt` (69 slides)
**Exam mapping:** Almost entirely **Exercise 1** (10 pts = 5+5)
**Priority:** HIGH for Ex 1, LOW for everything else

---

## Exam-Relevant Extractions

### 1. The Prof's Definition of ML (→ Exercise 1a, 5 pts)

The prof gives his own definition on slide 6:

> **"Machine learning is the science of fitting mathematical models to data"**

Expanded (slides 6, 37):

> Machine learning is to run computer algorithms on exemplary data to adjust the parameters of other computer algorithms (i.e. models) so that these become able to perform cognitive tasks.

**What to write on the exam for "What is Machine Learning?":**

1. State the core definition: ML = fitting mathematical models to data
2. Unpack: we have **data** (observations), a **model** (parameterized function), and a **fitting algorithm** (optimization) that adjusts model parameters so the model can perform tasks like prediction, classification, clustering
3. Mention the three ingredients: **data**, **model**, **optimization**
4. Mention generalization: the goal is not just to fit training data but to **generalize to unseen data** (slides 62-63)

> **Exam tip:** The prof's exact phrasing matters. Use "fitting mathematical models to data" — this is HIS definition, not the generic Tom Mitchell one. However, combining both is safest: ML = fitting models to data such that performance on tasks improves with experience.

---

### 2. Loss Function / Objective Function (→ Exercise 1b, 5 pts)

From slides 17, 23-24:

> A **loss function** $L(\theta | f_\theta, D)$ measures how well a model $f_\theta$ fits data $D$. The smaller the loss, the better the model fits.

Specific example from this lecture — the **mean squared error (MSE)** / **residual sum of squares (RSS)**:

$$E(\theta | f, D) = \frac{1}{n} \sum_{j=1}^{n} \left(f_\theta(x_j) - y_j\right)^2$$

And the **least squares loss**:

$$L(w) = \sum_{j=1}^{n} \left(f_\theta(x_j) - y_j\right)^2$$

From slide 24:
> Loss functions are **central to machine learning**. They are of the general form $L(\theta | f_\theta, D)$.

**What to write on the exam for "What is a loss function / objective function?":**

1. **Loss function:** $L(\theta | f_\theta, D)$ quantifies how poorly a model $f_\theta$ fits data $D$. Example: squared error $\ell(y, \hat{y}) = (y - \hat{y})^2$.
2. **Objective function:** the function we minimize (or maximize) to find optimal parameters. Often the aggregate loss over all data points.
3. **Role in ML:** The loss function turns the vague goal of "fit a model" into a precise **optimization problem**: $\hat{\theta} = \arg\min_\theta L(\theta | f_\theta, D)$. The choice of loss determines the behavior of the learned model.

> **This is where ML = optimization. The prof hammers this point: ML is model fitting, model fitting is optimization, and the loss function defines what "optimal" means.**

---

### 3. Key Concepts (Know They Exist — Low Detail Needed)

| Concept | Slide | Exam Use | What to Know |
|---------|-------|----------|--------------|
| Supervised vs Unsupervised | 16, 35-38 | Ex 1 background | Supervised = data has (x, y) pairs. Unsupervised = data has only x. |
| Regression | 20-25 | Background for Ex 2, 4 | Predict continuous y from x. |
| Classification | 27-29 | Background | Predict discrete label y from x. |
| Density estimation | 30-31 | Background | Estimate probability distribution from data. |
| Clustering | 33-34 | Background for k-means | Assign data to groups without labels. |
| Generalization | 62-63 | Ex 1 "why" questions | Good performance on **test** data (not training data) is what counts. $D_{trn} \cap D_{tst} = \emptyset$. |
| ML pipeline | 49, 53 | Not directly tested | Problem → Data → Preprocessing → Model selection → Fitting → Evaluation. |
| Feature extraction | 56-58 | Background | Transforming raw data into useful representations. |

---

### 4. Connections to Later Exercises

- **"Fitting models to data" → Exercise 2:** MLE/MAP are specific ways to fit models probabilistically
- **"Loss function" → Exercise 4:** The sum of squared distances is a loss function; minimizing it = the meta-pattern
- **"Generalization" → Exercise 5/7:** Kernels and GPs address overfitting/generalization
- **"Optimization" → Everything:** The prof introduces here that ML = optimization, which is the meta-pattern for the entire exam

---

## What to Skip

- **Historical remarks** (slide 3-4 about Gauss) — 0 exam points
- **ML pipeline stages in detail** (slides 47-66) — not tested as a standalone question
- **Foundation models / GenAI remarks** (slide 64) — not tested
- **Diagrams and visual examples** — useful for understanding, but the exam tests formulas

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 01:

1. **MSE:** $E(\theta) = \frac{1}{n}\sum_{j=1}^{n}(f_\theta(x_j) - y_j)^2$ → maps to **Ex 1b** (loss function definition)
2. **General loss:** $L(\theta | f_\theta, D)$ → maps to **Ex 1b**
3. **Optimization formulation:** $\hat{\theta} = \arg\min_\theta L(\theta)$ → maps to **Ex 1b, Ex 4**

**Verdict:** Know the prof's ML definition and the loss function concept cold. Everything else in this lecture is conceptual background — important for understanding but not directly tested as formulas. **Drill Ex 1 answers, then move on.**
