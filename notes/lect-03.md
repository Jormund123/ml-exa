# Lecture 03 — Gradient-Based Optimization, Gradient-Free Optimization, Gradient Flows

**Source:** `slides/lect-03.txt` (78 slides)
**Exam mapping:** Background for **Exercise 4** (optimization method), **Exercise 6b** (how to solve SVM), **Exercise 7b** (non-convex optimization — gradient descent with random restarts)
**Priority:** LOW for direct exam questions, but ONE concept from here is worth 3-5 points in Exercise 7b

---

## The ONE Thing That Matters for the Exam

### Gradient Descent for Non-Convex Problems (→ Exercise 7b, ~3-5 pts)

Exercise 7b asks: *"How would you solve the optimization problem, considering it is not convex?"*

**The answer comes from this lecture:**

> For non-convex problems (multiple local optima), use **gradient descent** (or gradient ascent for maximization) **with multiple random restarts**.
>
> 1. Initialize parameters randomly
> 2. Run gradient descent/ascent until convergence
> 3. Repeat from many different random starting points
> 4. Keep the best solution found

**Why random restarts?** Gradient descent converges to a LOCAL optimum. Different starting points may converge to different local optima. By restarting many times, we increase the chance of finding the GLOBAL optimum.

> **This is the answer pattern for any "non-convex" question on the exam.** If the exam asks "how would you solve X, considering it is not convex?" → gradient descent + multiple random restarts. Always.

---

## Gradient Descent — The Core Update Rule (know this conceptually)

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

- $\eta$ = step size / learning rate
- $\nabla f(x_k)$ = gradient at current point (direction of steepest increase)
- $-\nabla f(x_k)$ = negative gradient (direction of steepest DECREASE)

**Convex functions:** GD converges to the unique global minimum.
**Non-convex functions:** GD converges to a local minimum (depends on starting point).

---

## Optimization Methods Hierarchy (know names, not details)

| Method | Key Idea | Exam Relevance |
|--------|----------|----------------|
| GD (fixed step) | $x_{k+1} = x_k - \eta \nabla f(x_k)$ | Conceptual background |
| GD with line search | Optimize $\eta$ at each step | Not directly tested |
| Momentum GD | Average over past gradients, more robust | Not directly tested |
| ADAM | Best trade-off: stability + speed | Not directly tested |
| Newton's method | Uses Hessian (2nd order), few iterations but expensive | Not directly tested |
| Conjugate gradients | Converges in $n$ steps for $n$-dim convex quadratic | Not directly tested |
| Gradient-free (SPSA, Nelder-Mead) | No derivatives needed | Not directly tested |

**The prof's ranking (slide 49):**
- Stability → GD > Momentum > RMSprop > ADAM
- Speed → ADAM > RMSprop > Momentum > GD
- ADAM is most popular: best trade-off

> **For the exam:** You don't need to write ADAM's update equations. You need to know: (1) gradient descent exists, (2) for non-convex → multiple random restarts, (3) Frank-Wolfe for the SVM dual (that's in lect-11, not here).

---

## Gradient Flow (theoretical — likely not tested)

The continuous-time limit of gradient descent:

$$\dot{x}(t) = -\nabla E(x(t))$$

This is an ODE where the state evolves along the negative gradient of the energy. For convex quadratics, it converges exponentially to $x^* = (M^\top M)^{-1}M^\top v$.

**Exam relevance:** Near zero. This is elegant math but the exam tests formulas and derivations, not dynamical systems theory. **Skip for exam prep.**

---

## Connections to Exam Exercises

| Exercise | What This Lecture Contributes |
|----------|------------------------------|
| **Ex 4** | "How would you solve this?" → "Take derivative, set to zero" (closed-form possible here, so no iterative method needed) |
| **Ex 6b** | "How would you solve the SVM dual?" → Frank-Wolfe (from lect-11, not here) |
| **Ex 7b** | "How would you solve GP parameter optimization, considering it is not convex?" → **Gradient ascent + multiple random restarts** (THIS is the payoff from lect-03) |

---

## End-of-Lecture Summary

### Exam-relevant formulas from Lecture 03:

| # | Formula/Concept | Maps to |
|---|----------------|---------|
| 1 | $x_{k+1} = x_k - \eta \nabla f(x_k)$ (gradient descent) | **Ex 7b** (solving non-convex GP optimization) |
| 2 | Non-convex → multiple random restarts | **Ex 7b** (the key answer) |

### Verdict:

**One concept matters: "gradient descent with multiple random restarts for non-convex problems."** That's the answer to Exercise 7b's sub-question (~3-5 pts). Everything else in this lecture (ADAM, momentum, conjugate gradients, gradient flows) is background knowledge not directly examined.

**Drill this one-liner and move on.** Your time is better spent on lect-05 (30 pts of exam content) and lect-11 (25 pts of exam content).
