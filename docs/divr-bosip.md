# BOSIP-dIVR: Derivative-GP Integrated Variance Reduction

## 1. Setting

BOSIP performs **likelihood-based Bayesian inference**: given an observation $z_o$,
we seek the posterior over simulator parameters $x \in \mathcal{X} \subseteq \mathbb{R}^d$:

$$p(x \mid z_o) \propto p(z_o \mid f(x))\, p(x)$$

where $f: \mathcal{X} \to \mathbb{R}$ is an expensive simulator approximated by a
derivative-enhanced GP (see `docs/dkg-bosip.md §1–2` for the full setup).  After $n$
evaluations the GP posterior is:

$$f(x) \mid \mathcal{D}_n \sim \mathcal{N}\!\bigl(\mu_n(x),\, \sigma_n^2(x)\bigr)$$

and the current approximate posterior is:

$$q_n(x) \propto p(z_o \mid \mu_n(x))\, p(x)$$

---

## 2. Standard Integrated Variance Reduction (IVR)

For a plain GP, conditioning on a new scalar observation $f(x_\text{new})$ reduces
the posterior variance at every test point $x'$ **deterministically**:

$$\sigma_n^2(x') - \sigma_{n+1}^2(x') = \frac{\text{Cov}_n(f(x'), f(x_\text{new}))^2}{\text{Var}_n(f(x_\text{new}))}$$

The **Integrated Variance Reduction** acquisition integrates this reduction over the
domain, weighted by some importance measure $w(x')$:

$$\text{IVR}(x_\text{new}) = \int_\mathcal{X} \frac{c(x', x_\text{new})^2}{\sigma_n^2(x_\text{new})} \cdot w(x') \, dx'$$

IVR selects where to query next to most broadly reduce uncertainty — without the
mode-seeking bias of `max`-based acquisitions like Knowledge Gradient.

---

## 3. How the Derivative GP Changes IVR

### 3.1 Augmented observation

Each simulator call at $x_\text{new}$ returns the augmented vector:

$$\tilde{y}_\text{new} = \bigl[f(x_\text{new}),\; \partial f/\partial x_1(x_\text{new}),\; \ldots,\; \partial f/\partial x_d(x_\text{new})\bigr]^\top \in \mathbb{R}^{1+d}$$

This provides $1 + d$ virtual observations instead of 1.

### 3.2 Variance reduction formula

The deterministic variance reduction at $x'$ from observing $\tilde{y}_\text{new}$ generalises to a **quadratic form**:

$$\sigma_n^2(x') - \sigma_{n+1}^2(x') = c(x', x_\text{new})^\top \Sigma_n(x_\text{new})^{-1} c(x', x_\text{new}) = \|b(x', x_\text{new})\|^2$$

where:

- $c(x', x_\text{new}) \in \mathbb{R}^{1+d}$ — **posterior cross-covariance** between $f(x')$ and the augmented observation:
$$c_0 = \text{Cov}_n(f(x'),\, f(x_\text{new})), \qquad c_l = \text{Cov}_n(f(x'),\, \partial f/\partial x_l(x_\text{new}))$$

- $\Sigma_n(x_\text{new}) \in \mathbb{R}^{(1+d)\times(1+d)}$ — **posterior predictive covariance** of the augmented observation

- $b = L_\text{new}^{-1} c$ — **influence vector**, with $\Sigma_n = L_\text{new} L_\text{new}^\top$ (Cholesky)

### 3.3 Strict improvement over plain GP

The plain-GP reduction is recovered by ignoring gradient covariances:

$$\text{plain GP}: \quad \frac{c_0^2}{\Sigma_n[1,1]}$$

The derivative-GP reduction is:

$$\text{derivative GP}: \quad c^\top \Sigma_n^{-1} c = \|b\|^2 \geq \frac{c_0^2}{\Sigma_n[1,1]}$$

The inequality is strict whenever the gradient cross-covariances $c_1, \ldots, c_d$ are
non-zero and the gradient observations carry information beyond $f(x_\text{new})$ alone.
In practice this always holds for smooth kernels (Matérn, RBF) away from the boundary.

### 3.4 Trace form

The weighted integral over the domain can be written compactly as:

$$\text{IVR-dGP}(x_\text{new}) = \int_\mathcal{X} \|b(x', x_\text{new})\|^2 \cdot w(x') \, dx' = \operatorname{tr}\!\left[\Sigma_n(x_\text{new})^{-1}\, G(x_\text{new})\right]$$

where the **weighted information matrix** is:

$$G(x_\text{new}) = \int_\mathcal{X} c(x', x_\text{new})\, c(x', x_\text{new})^\top\, w(x') \, dx' \in \mathbb{R}^{(1+d)\times(1+d)}$$

---

## 4. BOSIP-dIVR: Posterior-Weighted Formulation

### 4.1 Posterior weight

For inference, variance reduction is only valuable where the posterior $q_n$ has mass.
We set:

$$w(x') = q_n(x') = \exp\!\bigl(\ell(\mu_n(x'),\, \sigma_n^2(x')) + \log p(x')\bigr)$$

where $\ell(\mu, \sigma^2) = \mathbb{E}_{y \sim \mathcal{N}(\mu,\sigma^2)}[\log p(z_o \mid y)]$
is the expected log-likelihood under the GP predictive (closed form for common
likelihoods; see `dkg-bosip.md §4.4`).

### 4.2 Monte Carlo approximation

The integral over $q_n$ is approximated by a Monte Carlo sum over a fantasy grid
$\{x'_1, \ldots, x'_M\}$ sampled from the prior and reweighted by the unnormalised
posterior:

$$\boxed{
\text{dIVR}(x_\text{new})
  = \sum_{m=1}^M \tilde{w}_m \cdot \|b_m\|^2,
  \qquad
  \tilde{w}_m = \frac{q_n(x'_m)}{\sum_{m'} q_n(x'_{m'})}
}$$

where $b_m = L_\text{new}^{-1} c(x'_m, x_\text{new})$.

### 4.3 Why dIVR does not cluster at the posterior mode

The key structural difference from dKG is the **sum** instead of **max**:

| Acquisition | Aggregation | Consequence |
|-------------|-------------|-------------|
| dKG         | $\max_m$    | $x_\text{new}$ chosen to improve the single best fantasy point → mode-seeking |
| dIVR        | $\sum_m \tilde{w}_m$ | $x_\text{new}$ chosen to reduce uncertainty across all regions where $q_n > 0$ |

With the sum, the acquisition landscape is smooth and free from winner-take-all
dynamics.  Points far from the mode contribute if they carry significant posterior
weight, naturally spreading simulator evaluations across the posterior support.

### 4.4 Connection to LogMaxVar

`LogMaxVar` also avoids mode-seeking but selects the single point of maximum
posterior-weighted variance:

$$\text{LogMaxVar}(x_\text{new}) = \max_{x'} \sigma_n^2(x')\, q_n(x')$$

dIVR is preferable when you want to account for the global information gain of
observing $(f, \nabla f)$ jointly, rather than the marginal variance at a single point.

---

## 5. Algorithm

```
Input:  current GP posterior (μₙ, σₙ, chol),  bosip.likelihood,  bosip.x_prior
        fantasy grid {x'₁, …, x'_M}  (uniform random over domain, fixed per step)

Precompute (O(M · N) — done once per BO step):
  μ_grid[m]    = μₙ(x'_m)                                 for each m
  σ²_grid[m]   = σ²ₙ(x'_m)                                for each m
  a[m]         = ℓ(μ_grid[m], σ²_grid[m]) + log p(x'_m)  for each m
  w̃[m]         = softmax(a)[m]                             (log-sum-exp stable)
  alpha_star_m = K_aug⁻¹ k_cross(x'_m)                    for each m

For each candidate x_new (O(M · (1+d)²) per candidate):
  1. Σₙ(x_new):  (1+d)×(1+d) posterior predictive covariance
  2. L_new = chol(Σₙ(x_new) + ε I)
  3. K_cross_new = _build_cross_cov_matrix(k_fn, x_new, X_train)   (N × 1+d)
  4. total = 0
     For m = 1…M:
       c_m  = k_prior(x'_m, x_new) - K_cross_new' alpha_star_m     (1+d vector)
       b_m  = L_new \ c_m                                           (1+d vector)
       bsq  = min(‖b_m‖², σ²_grid[m])                              (clip invariant)
       total += w̃[m] * bsq
  5. dIVR(x_new) = total

Output:  x*_new = argmax_{x_new} dIVR(x_new)
```

**No Monte Carlo loop** — the acquisition is fully deterministic.

---

## 6. Computational Complexity

Per BO step:

| Step | Cost |
|------|------|
| Grid precomputation (μ, σ², weights) | $O(M \cdot N)$ |
| Cholesky solve per grid point | $O(M \cdot N)$ |
| `_build_cross_cov_matrix` per candidate | $O(n \cdot (1+d)^2)$ |
| Influence vector loop | $O(M \cdot (1+d)^2)$ |
| **Total per candidate** | $O(M \cdot N)$, $N = n(1+d)$ |

Compared to dKG ($O(S \cdot M \cdot (1+d))$ per candidate with $S$ MC samples),
dIVR is cheaper by a factor of $S$ while being fully deterministic.

---

## 7. Numerical Note: Influence Vector Clipping

Near existing training points, catastrophic cancellation in the posterior cross-covariance
$c_m$ can inflate $b_m$ numerically, violating the mathematical invariant
$\|b_m\|^2 \leq \sigma_n^2(x'_m)$.  The implementation clips:

$$\|b_m\|^2_\text{used} = \min\!\bigl(\|b_m\|^2,\; \sigma_n^2(x'_m)\bigr)$$

This enforces the invariant (variance reduction cannot exceed existing variance) and
prevents the acquisition from peaking artificially at already-observed locations.

---

## 8. References

- **Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989).**
  *Design and Analysis of Computer Experiments.*
  Statistical Science, 4(4).

- **Lam, C. Q. (2008).**
  *Sequential Adaptive Designs in Computer Experiments for Response Surface Model Fit.*
  PhD thesis, Ohio State University.
  *(Integrated Mean Squared Error / IVR for computer experiments.)*

- **Wu, J., Poloczek, M., Wilson, A. G., & Frazier, P. I. (2017).**
  *Bayesian Optimization with Gradients.*
  NeurIPS 30.
  *(Derivative GP kernel and influence vector formulation.)*

- **Järvenpää, M., Gutmann, M. U., Vehtari, A., & Marttinen, P. (2021).**
  *Parallel Gaussian Process Surrogate Bayesian Inference with Noisy Likelihood
  Evaluations.*
  Bayesian Analysis, 16(1).
  *(BOSIP framework and posterior-weighted acquisitions.)*
