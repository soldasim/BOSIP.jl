# BOSIP-dKG: Derivative Knowledge Gradient for Likelihood-Based Inference

## 1. Setting

BOSIP performs **likelihood-based Bayesian inference**: given a scalar or vector
observation $z_o$ from a real experiment, we seek the posterior over simulator
parameters $x \in \mathcal{X} \subseteq \mathbb{R}^d$:

$$p(x \mid z_o) \propto p(z_o \mid f(x))\, p(x)$$

where $f: \mathcal{X} \to \mathbb{R}^{d_y}$ is an expensive simulator.
The likelihood $p(z_o \mid y)$ translates simulator output $y = f(x)$ into a
probability of the observed data.

Because $f$ is expensive, it is approximated by a **Gaussian process surrogate**
trained on evaluations $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$.  After $n$
evaluations the GP posterior is:

$$f(x) \mid \mathcal{D}_n \sim \mathcal{N}\!\bigl(\mu_n(x),\, \sigma_n^2(x)\bigr)$$

and the approximate log-posterior used for inference is:

$$\log q_n(x) = \log p\!\bigl(z_o \mid \mu_n(x)\bigr) + \log p(x)$$

---

## 2. Derivative-Enhanced GP (Wu et al., 2017)

Each simulator call returns **both** the function value and its gradient via
automatic differentiation:

$$f(x_\text{new}) \;\longrightarrow\; \bigl(y_\text{new},\; \nabla y_\text{new}\bigr)$$

The augmented observation vector is

$$\tilde{y}_\text{new} = \bigl[y_\text{new},\; \partial y/\partial x_1,\; \ldots,\; \partial y/\partial x_d\bigr]^\top \in \mathbb{R}^{1+d}$$

The GP is extended to jointly model $f$ and $\partial f / \partial x_l$ via the
**derivative kernel** blocks (Wu et al., 2017, §2):

$$\begin{aligned}
k\!\bigl(f(x_i),\, f(x_j)\bigr) &= k(x_i, x_j) \\
k\!\bigl(f(x_i),\, \partial f(x_j)/\partial x_l^j\bigr) &= \partial k(x_i, x_j)/\partial x_l^j \\
k\!\bigl(\partial f(x_i)/\partial x_l^i,\, \partial f(x_j)/\partial x_m^j\bigr)
  &= \partial^2 k(x_i,x_j)/(\partial x_l^i\, \partial x_m^j)
\end{aligned}$$

Each evaluation provides $1 + d$ "virtual" observations instead of 1, accelerating
posterior concentration.

### One-Step Posterior Mean Update

After adding a new augmented observation $\tilde{y}_\text{new}$ at $x_\text{new}$,
the GP posterior mean at any test point $x'$ updates as:

$$\mu_{n+1}(x') = \mu_n(x') + c(x', x_\text{new})^\top \Sigma_n(x_\text{new})^{-1}
  \bigl(\tilde{y}_\text{new} - \tilde{\mu}_n(x_\text{new})\bigr)$$

where:

- $c(x', x_\text{new}) \in \mathbb{R}^{1+d}$ is the **posterior cross-covariance**
  between $f(x')$ and the augmented observation:
  $$c_0 = \text{Cov}_n\!\bigl(f(x'),\, f(x_\text{new})\bigr),\quad
    c_l = \text{Cov}_n\!\bigl(f(x'),\, \partial f(x_\text{new})/\partial x_l\bigr)$$

- $\Sigma_n(x_\text{new}) \in \mathbb{R}^{(1+d)\times(1+d)}$ is the **posterior
  predictive covariance** of the augmented observation at $x_\text{new}$:
  $$\Sigma_n(x_\text{new}) = K_\text{aug}(x_\text{new},x_\text{new})
    - K_\text{cross}^\top K_\text{aug}^{-1} K_\text{cross} + \text{diag}(\sigma^2,\sigma_\partial^2\mathbf{1})$$

Since $\tilde{y}_\text{new} - \tilde{\mu}_n(x_\text{new}) \sim \mathcal{N}(0, \Sigma_n)$,
we write $\Sigma_n = L L^\top$ (Cholesky) and parameterize the innovation as
$L\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I_{1+d})$:

$$\boxed{\mu_{n+1}(x') = \mu_n(x') + b(x')^\top \varepsilon}$$

where the **influence vector** is:

$$b(x') = L^{-\top} c(x', x_\text{new}) \in \mathbb{R}^{1+d}$$

---

## 3. Standard dKG (GP Optimisation)

For plain Bayesian optimisation (maximise $f$ directly), the **discrete dKG** value
at candidate $x_\text{new}$ is (Wu et al., 2017, §3):

$$\text{dKG}(x_\text{new})
  = \mathbb{E}_\varepsilon\!\left[\max_{m} \bigl(\mu_n(x'_m) + b_m^\top\varepsilon\bigr)\right]
  - \max_m \mu_n(x'_m)$$

where $\{x'_m\}_{m=1}^M$ is a discrete **fantasy grid** over $\mathcal{X}$, and
$b_m = b(x'_m)$.

This is estimated by Monte Carlo:

$$\text{dKG}(x_\text{new}) \approx \frac{1}{S}\sum_{s=1}^S
  \max_m\!\bigl[\mu_n(x'_m) + b_m^\top\varepsilon_s\bigr]
  - \max_m \mu_n(x'_m), \quad \varepsilon_s \overset{\text{iid}}{\sim} \mathcal{N}(0,I)$$

This formulation is **correct for direct GP optimisation** but does not account
for the likelihood transformation in BOSIP.

---

## 4. Likelihood-Aware Extension (BOSIP-dKG)

In BOSIP the objective is not $f(x')$ itself but the **log-posterior**
$\log q(x') = \log p(z_o \mid f(x')) + \log p(x')$.

### 4.1 Expected Log-Likelihood Under the GP Predictive

Because the GP posterior variance $\sigma_n^2(x')$ is non-zero, the simulator
output at $x'$ is uncertain.  Rather than plugging in the mean, we marginalise
over the GP predictive distribution by defining the **expected log-likelihood**:

$$\ell(\mu, \sigma^2) = \mathbb{E}_{y \sim \mathcal{N}(\mu, \sigma^2)}\!\bigl[\log p(z_o \mid y)\bigr]$$

This has closed forms for common likelihoods (see §4.3).

### 4.2 One-Step Variance Reduction

After observing the augmented $(f, \nabla f)$ at $x_\text{new}$, the GP
posterior variance at each fantasy point $x'_m$ **deterministically decreases**:

$$\sigma_{n+1}^2(x'_m) = \sigma_n^2(x'_m) - \|b_m\|^2$$

where $b_m = L^{-1} c(x'_m, x_\text{new})$ is the influence vector.  This
reduction is exact (not stochastic) and must be included for the acquisition to
be positive: for concave likelihoods (e.g. Normal), ignoring the variance
reduction yields $\mathbb{E}[\log p(z_o \mid \mu + \delta)] \le \log p(z_o \mid \mu)$
by Jensen's inequality, making dKG incorrectly negative everywhere.

### 4.3 BOSIP-dKG Formula

We define the BOSIP-dKG as the expected improvement in the **best achievable
expected log-posterior** over the fantasy grid:

$$\text{dKG}_\text{BOSIP}(x_\text{new})
  = \mathbb{E}_\varepsilon\!\left[
      \max_m \bigl(\ell(\mu_{n+1}(x'_m),\, \sigma_{n+1}^2(x'_m)) + \log p(x'_m)\bigr)
    \right]
  - \max_m \bigl(\ell(\mu_n(x'_m),\, \sigma_n^2(x'_m)) + \log p(x'_m)\bigr)$$

Substituting the one-step updates:

$$\boxed{
\text{dKG}_\text{BOSIP}(x_\text{new})
  = \frac{1}{S}\sum_{s=1}^S
    \max_m\!\Bigl[\ell\!\bigl(\mu_n(x'_m) + b_m^\top\varepsilon_s,\; \sigma_n^2(x'_m) - \|b_m\|^2\bigr)
                  + \log p(x'_m)\Bigr]
  - \max_m\!\Bigl[\ell\!\bigl(\mu_n(x'_m),\; \sigma_n^2(x'_m)\bigr) + \log p(x'_m)\Bigr]
}$$

### 4.4 Likelihood-Specific Forms

The BOSIP-dKG formula is generic in the likelihood.  Two important special cases:

#### Normal likelihood

The GP models the simulator output $y$ directly.  Marginalising over the GP
predictive $y \sim \mathcal{N}(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = \log \mathcal{N}(z_o;\; \mu,\; \sigma^2 + \sigma_\text{obs}^2)
  = -\frac{(z_o - \mu)^2}{2(\sigma^2 + \sigma_\text{obs}^2)}
    - \tfrac{1}{2}\log\!\bigl(2\pi(\sigma^2+\sigma_\text{obs}^2)\bigr)$$

This is the analytic Gaussian convolution: $z_o \mid x \sim \mathcal{N}(\mu, \sigma^2 + \sigma_\text{obs}^2)$.

The resulting $\ell(\mu_{n+1}(x'), \sigma_{n+1}^2(x'))$ is a **nonlinear**
function of the stochastic update $\delta = b^\top\varepsilon$, so Monte Carlo
is required.

#### Exponential likelihood (log-likelihood surrogate)

The GP models the log-likelihood directly: $f(x) \approx \log p(z_o \mid x)$.
The likelihood is $p(z_o \mid y) = \exp(y)$, so:

$$\ell(\mu, \sigma^2)
  = \mathbb{E}_{y \sim \mathcal{N}(\mu,\sigma^2)}[y]
  = \mu + \tfrac{\sigma^2}{2}$$

(using the log-expectation of a lognormal).  Substituting the one-step update:

$$\ell(\mu_n(x') + \delta,\; \sigma_n^2 - \|b\|^2)
  = \mu_n(x') + \delta + \tfrac{1}{2}(\sigma_n^2 - \|b\|^2)$$

The expectation over $\varepsilon$ of the **linear** term $\delta = b^\top\varepsilon$
is zero, so the improvement comes entirely from the **variance reduction term**
$-\|b\|^2/2$.  BOSIP-dKG with `ExpLikelihood` therefore selects the point that
maximally reduces the lognormal mean variance — a pure exploration strategy.

---

## 5. Algorithm

```
Input:  current GP posterior (μₙ, σₙ, chol),  bosip.likelihood,  bosip.x_prior
        fantasy grid {x'₁, …, x'_M}  (uniform random over domain)

Precompute:
  μ_grid[m]    = μₙ(x'_m)                               for each m
  σ²_grid[m]   = σ²ₙ(x'_m)                              for each m
  log_prior[m] = log p(x'_m)                             for each m
  a[m]         = ℓ(μ_grid[m], σ²_grid[m]) + log_prior[m]    [using _ell dispatch]
  best_a       = max_m a[m]
  alpha_star_m = K_aug⁻¹ k_cross(x'_m)                  for each m   (one Cholesky solve)

For each candidate x_new:
  1. Compute Σₙ(x_new):  (1+d)×(1+d) posterior predictive covariance
                          = K_aug_prior(x_new) - V'V,  V = L_train \ K_cross_new
  2. L_new = chol(Σₙ(x_new))
  3. K_cross_new = _build_cross_cov_matrix(k_fn, x_new, X_train)   (N × 1+d)
  4. For m = 1…M:
       c_m           = k_prior(x'_m, x_new) - K_cross_new' alpha_star_m    (1+d vector)
       b_m           = L_new \ c_m                                          (1+d vector)
       σ²_updated[m] = max(0, σ²_grid[m] - ‖b_m‖²)              [deterministic reduction]
  5. expected_max = 0
     For s = 1…S:
       ε_s ~ N(0, I_{1+d})
       δ   = B' ε_s                                               (M vector)
       expected_max += max_m [ ℓ(μ_grid[m] + δ[m], σ²_updated[m]) + log_prior[m] ]
  6. dKG(x_new) = expected_max / S - best_a

Output:  x*_new = argmax_{x_new} dKG(x_new)
```

---

## 6. Computational Complexity

Per BO step (candidate evaluation):

| Step | Cost |
|------|------|
| Fantasy grid precomputation | $O(M \cdot n \cdot (1+d))$ |
| Cholesky solve per fantasy point | $O(M \cdot N)$, $N = n(1+d)$ |
| `_build_cross_cov_matrix` per candidate | $O(n \cdot (1+d)^2)$ |
| Influence matrix $B$ | $O(M \cdot (1+d)^2)$ |
| MC loop | $O(S \cdot M \cdot (1+d))$ |

For the 1D example ($n = 10$, $d = 1$, $M = 64$, $S = 512$): roughly $10^5$
floating-point operations per candidate.  With 24 multistarts for the outer
optimizer (BOBYQA), one acquisition step takes $\sim 24 \times 10^5 \approx 2.4\text{M}$
operations, which completes in well under a second.

---

## 7. References

- **Wu, J., Poloczek, M., Wilson, A. G., & Frazier, P. I. (2017).**
  *Bayesian Optimization with Gradients.*
  Advances in Neural Information Processing Systems (NeurIPS) 30.

- **Scott, W., Frazier, P. I., & Powell, W. B. (2011).**
  *The Correlated Knowledge Gradient for Simulation Optimization of Continuous
  Parameters using Gaussian Process Regression.*
  SIAM Journal on Optimization, 21(3).

- **Järvenpää, M., Gutmann, M. U., Vehtari, A., & Marttinen, P. (2021).**
  *Parallel Gaussian Process Surrogate Bayesian Inference with Noisy Likelihood
  Evaluations.*
  Bayesian Analysis, 16(1).
