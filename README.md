# Linear Regression from Scratch (NumPy only)

A from-scratch implementation of linear regression — gradient descent, MSE loss,
L2 (Ridge) regularization, and the closed-form Normal Equation — built using
only NumPy. `scikit-learn` is used strictly for data loading and final
validation, never for modeling.

## Why

Before relying on `model.fit()`, I wanted to rebuild the mechanics underneath
it: the loss function, the gradients, and the update rule, from first
principles. This is Project 1 in a self-directed ML-from-scratch sequence.

## What's implemented

- Manual train/test split and feature standardization (no `sklearn` helpers)
- `LinearRegressionGD`: batch gradient descent with optional L2 regularization
- Closed-form Normal Equation solver
- Validation on a synthetic dataset (`y = 3x + 5 + noise`) and the real-world
  California Housing dataset

## Results

### Learning rate behavior (synthetic data)

I swept learning rates from 0.001 to 1.0 to map out the full convergence
spectrum:

| Learning rate | Behavior |
|---|---|
| 0.001 – 0.01 | Converges, but slowly — still descending after 200 epochs |
| 0.02 – 0.5 | Converges cleanly, faster as LR increases, no visible oscillation |
| 0.6 – 0.9 | Converges to the same optimal loss (~0.878), with negligible drift at the upper end |
| 1.0 | Diverges — loss stays flat at the initial value for the entire run |

The divergence at `lr=1.0` isn't a runaway explosion to infinity — it's a
stable 2-cycle: the update overshoots the minimum, lands on a mirror-image
point with identical loss (since MSE is a symmetric quadratic), then
overshoots back. The loss trace looks flat because both points in the cycle
have the same loss, even though the weights are still bouncing between them
each epoch.

The stability boundary sits somewhere between `lr=0.9` and `lr=1.0` —
consistent with the theoretical gradient descent stability condition
`lr < 2/L`, where `L` is tied to the curvature of the loss surface.

![Loss curves across learning rates](path/to/lr_comparison.png)

### Regularization on vs. off (California Housing)

| | alpha=0.0 | alpha=1.0 |
|---|---|---|
| Test MSE | 0.5249 | 0.8172 |
| Weight L2 norm | ~1.42 | ~0.42 |

At `alpha=1.0`, weights shrink substantially (some by >80%), but test MSE
*increases* rather than decreases — this regularization strength is too
aggressive for this dataset and underfits rather than helps generalization.
This is a useful negative result: regularization strength is a
hyperparameter to tune, not something that monotonically improves
performance. A smaller alpha (e.g. 0.01–0.1) would likely strike a better
bias-variance tradeoff.

### Convergence validation: Gradient Descent vs. Normal Equation

| Method | Time | Weight diff vs Normal Eq | Bias diff |
|---|---|---|---|
| Normal Equation | 0.00393s | — | — |
| Gradient Descent | 0.16466s | 6.30e-05 | 1.15e-14 |

GD converges to within ~6e-5 of the exact closed-form solution — at ~40x the
compute cost. This is the expected tradeoff: the Normal Equation is fast and
exact for small/medium feature counts, but doesn't scale to high dimensions
or to problems with no closed form (e.g. logistic regression). GD
generalizes; the Normal Equation doesn't.

### Validation against scikit-learn

| | My model | scikit-learn |
|---|---|---|
| R² (test) | 0.6100490 | 0.6100480 |
| Weight diff (L2 norm) | 6.30e-05 | — |
| Bias diff | 8.88e-16 | — |

Agreement to 5 decimal places confirms the implementation is correct.

> Note: R² ≈ 0.61 reflects the known limits of *linear* regression on this
> dataset (nonlinear feature relationships, target clipping at $500k) — the
> goal here was implementation correctness, not predictive performance.
