# POML Exam Strategy Guide

Instructions for studying the "Principles of Machine Learning" (MA-INF 4111) exam.
Use this as a companion while reading each lecture slide. Feed this to your AI agent for context.

---

## Exam Structure (from past papers)

Past exams 1A and 1B (our course) had the **exact same 7-question structure**, same topics, same order, same point distribution. A third paper (1C) is from a different course (Uni Luebeck, Prof. Rueckert, "Probabilistic ML") but has significant topic overlap — see notes below the table.

| # | Topic | Points | Source Lectures |
|---|-------|--------|-----------------|
| 1 | Definitions (what is ML, what is a loss function) | 10 | lect-00, lect-05 |
| 2 | Likelihood (joint prob, log-likelihood, MLE, MAP) | 10 | lect-05, lect-06 |
| 3 | Bayesian inference (Bayes theorem, conjugate priors, posterior predictive, why it's hard) | 10 | lect-05, lect-06 |
| 4 | Simple convex optimization (argmin of sum of squared distances) | 10 | lect-09 |
| 5 | Kernel trick (what is it, where is it used) | 10 | lect-11, lect-12 |
| 6 | L2 SVM (dual form, how to solve, recover w and b) | 15 | lect-11 |
| 7 | Gaussian processes (non-zero mean handling, kernel parameter optimization, non-convex) | 15 | lect-12, (lect-07) |
| | **Total** | **80** | Pass = 40 |

**Key insight:** Exercises 6 and 7 are worth 30/80 combined. They are the exam's center of gravity.

### Exam 1C Cross-Reference (different course, overlapping topics)

**Source:** `past-questions/POML_2324_WS_1C.txt` — Uni Luebeck, Prof. Rueckert, 4 questions × 25 pts = 100 pts, 90 min, aids allowed.

| 1C Question | Overlapping Topic | Reinforces Our Exercise |
|-------------|-------------------|------------------------|
| Q1c: Apply Bayes' theorem to Hepatitis B test | Bayes' theorem with concrete numbers | **Ex 3a** (write Bayes, name all parts) |
| Q1d: Beta-Binomial conjugacy (casino cheating) | Conjugate priors worked example | **Ex 3b** (explain conjugate priors) |
| Q2b-c: Derive LSQ solution + ridge regression | $\mu = (K^\top K + \lambda I)^{-1}K^\top g$ | **Ex 4** (meta-pattern), **Ex 3** (MAP = regularization) |
| Q3a: Identify mistakes in GP sketch | GP mean/variance behavior | **Ex 7** (GP understanding) |
| Q3b: True/False about GPs | Kernel matrix must be positive definite; GP is $O(N^3)$ | **Ex 7** (reasoning questions) |
| Q3c: Match kernels to GP sample plots | Kernel parameters control function smoothness | **Ex 5, 7** (kernel understanding) |
| Q4b: Derive posterior predictive via marginalization | $p(\tau) = \int p(\tau|\theta)p(\theta)d\theta$ | **Ex 3c** (posterior predictive) |

**Key 1C takeaways for our exam:**
- **Concrete conjugate prior example:** Beta prior + Binomial likelihood → Beta posterior. Use this if Ex 3b asks for an example.
- **Ridge regression = MAP with Gaussian prior:** $L_r = \frac{1}{2}w^\top w$ ↔ $p(w) = \mathcal{N}(0, \frac{1}{\lambda}I)$. If a "why regularization" question appears, state this connection.
- **GP kernel matrix must be positive definite.** GP cost = $O(N^3)$ from inverting $N \times N$ covariance matrix. Variance is low near data, high far from data. These are "reason about" points for Ex 7.

---

## The ONE Meta-Pattern

Nearly every derivation in this course follows the same skeleton:

```
1. OBJECTIVE: write down what to minimize/maximize
2. CONSTRAINTS? → build a Lagrangian (objective + lambda * constraint)
3. DIFFERENTIATE: take partial derivatives, set each = 0 (KKT conditions)
4. SOLVE: the resulting system of equations
```

When reading ANY slide that shows a derivation, identify which step you're looking at. The prof always follows this pattern. If you internalize it, you can reconstruct derivations you've never seen.

---

## How to Read Each Lecture

### lect-00 to lect-02: Foundations (Introduction + Regression)
- **What to extract:** The definition of ML from the prof's own words (lect-01 slide 6: "ML is the science of fitting mathematical models to data"). Least squares loss, feature maps $\phi(x)$, feature matrix $\Phi$, the closed-form solution $\hat{w} = (\Phi\Phi^\top)^{-1}\Phi y$.
- **Exam relevance:** Exercise 1 (definitions), plus foundational notation used everywhere.
- **Skip:** Python code examples, historical remarks, height/weight plot details.

### lect-03: Gradient-Based Optimization
- **What to extract:** Gradient descent update rule $x_{k+1} = x_k - \eta\nabla f(x_k)$. For non-convex problems → gradient descent with **multiple random restarts**.
- **Exam relevance:** The "multiple random restarts" answer is worth 3-5 pts in Exercise 7b. Everything else (ADAM, momentum, conjugate gradients, gradient flows) is background.
- **Skip:** Detailed proofs of convergence, gradient flow ODEs, Nelder-Mead.

### lect-04: Binary Classification, Logistic Regression, Autodiff
- **What to extract:** Linear binary classifier $f(x) = \text{sign}(x^\top w - b)$, bipolar labels $y_j \in \{-1,+1\}$, logistic loss (convex). Regularization term $L_r = \frac{1}{2}w^\top w$ foreshadows MAP = ridge regression.
- **Exam relevance:** Exercise 1 (secondary loss function example), notation background for Exercise 6 (SVM). Low priority overall.
- **Skip:** ALL Python/JAX code, autodiff theory, computation graphs — 0 exam points.

### lect-05: Probabilistic Model Fitting (Part 1)
- **This is the single most tested lecture.** Exercises 1, 2, and 3 all draw from it.
- **What to extract:** The 4-formula chain: joint probability (product) -> log-likelihood (sum) -> MLE (argmax likelihood) -> MAP (argmax posterior). Also: posterior predictive distribution definition, conjugate prior definition, and the discussion of why Bayesian inference is hard.
- **Pay attention to:** The specific notation the prof uses. The exam gives you blank formulas to fill in using his notation.

### lect-06 / lect-07: Probabilistic Model Fitting (Part 2)
- **What to extract:** Conjugate priors worked examples, posterior predictive in practice, Bayesian regression, connection between regularization and MAP.
- **Exam relevance:** Exercise 3 (explain conjugate priors, write posterior predictive, reason about why Bayes is hard). The prof lists advantages and disadvantages of Bayesian estimation here - memorize those bullet points.
- **Key connection:** MAP estimation with Gaussian prior = ridge regression (L2 regularization). This shows up as a "why" question.

### lect-08: Neural Networks / Normalization
- **What to extract:** Data normalization / standardization (centering to zero mean). This concept reappears in Exercise 7 (GP with non-zero mean) and PCA.
- **Lower exam priority** unless PCA or normalization appears as a new question.

### lect-09: Constrained Optimization
- **This teaches the meta-pattern.** Read it to understand Lagrange multipliers and KKT conditions.
- **What to extract:** The Lagrangian construction, the DTMC example (constrained MLE of a transition matrix), and the general principle of converting constrained -> unconstrained problems.
- **Exam relevance:** Exercise 4 (simple optimization) and Exercise 6 (SVM dual derivation via KKT). Also the mindset for Exercise 7.

### lect-10: SVM Part 1
- **What to extract:** Why SVMs maximize the margin, what support vectors are, the primal formulation, slack variables.
- **Exam relevance:** Background for Exercise 6. You need to understand the primal to appreciate the dual.

### lect-11: SVM Part 2 (CRITICAL)
- **This is the highest-ROI lecture for the exam.** Exercise 6 (15 pts) comes directly from here.
- **What to extract:** Three SVM variants (L1, Least Squares, L2) and their primal/dual forms. The exam asks specifically about the **L2 SVM dual**. Trace the full derivation: primal -> Lagrangian -> KKT -> eliminate variables -> dual. Also memorize the formulas for recovering w and b from the KKT conditions.
- **Also here:** The kernel trick definition (2 steps), kernel engineering, and kernel SVMs. This feeds Exercise 5.
- **The Frank-Wolfe algorithm** is mentioned as the solver for the L2 dual. Know it exists and what it does (iterative, linearize, step toward simplex vertex).

### lect-12: Kernelized Regression, Overfitting, GPs
- **What to extract:** Kernel least squares regression formula, the connection between kernel regression and Gaussian process regression (they're the same!), overfitting discussion.
- **Exam relevance:** Exercise 7 draws from the GP regression content. Also supports Exercise 5 (kernel trick applications).

### lect-13: PCA / Kernel PCA
- **What to extract:** Spectral decomposition, eigenvectors = principal components, centering matrix, kernel PCA via Gram matrix.
- **Exam relevance:** Not directly tested in past exams, but could appear as a new Exercise 4/5 variant. The centering concept connects to Exercise 7 (centering y for GPs).
- **If you're short on time:** know what PCA does conceptually and that kernel PCA replaces the covariance matrix with a kernel matrix.

### lect-14: K-Means, GMMs, EM Algorithm
- **What to extract:** K-means objective, EM algorithm (E-step = soft assignment, M-step = update parameters), GMM as "soft k-means".
- **Exam relevance:** Not directly in past exams but could replace any exercise. The k-means centroid update is the same as Exercise 4's answer (sample mean). Know EM at a conceptual level.

### lect-15: ITVQ, RKHS, Mean Discrepancy
- **What to extract:** Parzen windows, RKHS theory, maximum mean discrepancy.
- **Exam relevance:** Lowest priority. The prof calls it "extremely dense" and it's the finale. Unlikely to be tested in detail, but kernel concepts from here reinforce Exercise 5.

---

## Patterns Observed From Past Exams

1. **1A and 1B are near-identical.** Both test the same topics with the same structure. Expect this year's exam to follow the same template. 1C (different course) confirms that Bayes, GPs, ridge regression, and kernel functions are universally tested across ML exams.

2. **Question types map to specific verbs:**
   - "Write down..." = reproduce a specific formula from slides
   - "Derive..." / "Compute..." = apply the meta-pattern (objective -> derivative -> solve)
   - "Explain..." / "What is..." = give the definition + why it matters
   - "How would you solve..." = name the algorithm/method, explain the steps
   - "Reason about why..." = give 2-3 arguments with keywords

3. **The exam builds a chain:** Ex 1 (definitions) -> Ex 2 (likelihood formulas) -> Ex 3 (Bayesian layer on top) -> Ex 4 (optimization fundamentals) -> Ex 5 (kernel trick) -> Ex 6 (SVM uses optimization + kernels) -> Ex 7 (GP uses kernels + likelihood + optimization). Later exercises assume earlier ones. If you understand the chain, unseen variations become manageable.

4. **Partial credit strategy:** The exam has many sub-parts (a, b, c, d). Each sub-part is independently graded. Even if you can't finish a derivation, writing the objective and the first derivative gets you points. Never skip a sub-part.

5. **The 40-point pass line** means you can miss up to 40 points and still pass. Exercises 1-4 (40 pts) are highly memorizable. Nail those and you pass.

---

## Strategy for Unseen Problems

The exam might change a question. Here's how to handle it:

**If it's an optimization problem you've never seen:**
-> Write the objective. Check for constraints. Apply the meta-pattern. Even if you can't solve the algebra, the setup earns points.

**If it asks about a model you recognize but the question is new:**
-> Connect it back to something you know. Everything in this course connects: regression = MLE, regularization = MAP, kernel regression = GP, k-means centroid = sample mean. State the connection.

**If it asks "why" or "explain" about something unfamiliar:**
-> Use the course's recurring themes as keywords: convexity, overfitting, generalization, tractability, regularization, kernel trick, i.i.d. assumption.

**If it involves inner products $x_i^\top x_j$ anywhere:**
-> Mention the kernel trick. This is almost always relevant.

**If it's about solving a non-convex problem:**
-> Answer: gradient descent with multiple random restarts (to escape local optima). This applies to GPs, neural networks, and EM.

---

## Lecture Priority (time-constrained order)

If you only have time for some lectures, read them in this order:

1. **lect-11** (SVM Part 2) - 15 pts directly, also covers kernel trick (10 pts) = 25 pts
2. **lect-05** (Probabilistic Model Fitting 1) - covers Ex 1, 2, 3 = 30 pts
3. **lect-09** (Constrained Optimization) - teaches the meta-pattern, covers Ex 4 = 10 pts
4. **lect-06** (Probabilistic Model Fitting 2) - deepens Ex 3, Bayesian regression
5. **lect-12** (Kernelized Regression, GPs) - covers Ex 7 = 15 pts
6. **lect-14** (K-Means, GMMs, EM) - wildcard insurance
7. **lect-13** (PCA) - wildcard insurance
8. Everything else
