# Derivative-GP Acquisition Functions for BOSIP

This document covers how the BOSIP acquisition functions are modified when the
surrogate is a **derivative-enhanced GP** (`GradientGaussianProcess`) that jointly
observes function values and gradients.

---

## 1. Setting

BOSIP performs **likelihood-based Bayesian inference**: given an observation $z_o$,
we seek the posterior over simulator parameters $x \in \mathcal{X} \subseteq \mathbb{R}^d$:

$$p(x \mid z_o) \propto p(z_o \mid f(x))\, p(x)$$

where $f: \mathcal{X} \to \mathbb{R}$ is an expensive simulator approximated by a GP
surrogate trained on evaluations $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$.  The GP
posterior is:

$$f(x) \mid \mathcal{D}_n \sim \mathcal{N}\!\bigl(\mu_n(x),\, \sigma_n^2(x)\bigr)$$

and the current approximate posterior is $q_n(x) \propto p(z_o \mid \mu_n(x))\, p(x)$.

---

## 2. Derivative-Enhanced GP

Each simulator call returns the augmented observation:

$$\tilde{y}_\text{new} = \bigl[f(x_\text{new}),\; \partial f/\partial x_1(x_\text{new}),\; \ldots,\; \partial f/\partial x_d(x_\text{new})\bigr]^\top \in \mathbb{R}^{1+d}$$

The GP is extended via derivative kernel blocks so that function values and gradients
are jointly modelled (Wu et al., 2017).  After conditioning on $n$ evaluations the
posterior mean and variance are computed as before, but with $n(1+d)$ effective
observations instead of $n$.

---

## 3. Generic One-Step Posterior Update

This section derives the update equations used by all lookahead acquisitions.

### 3.1 Influence vector

Given a candidate $x_\text{new}$, define:

- $\Sigma_n(x_\text{new}) \in \mathbb{R}^{(1+d)\times(1+d)}$ — **posterior predictive
  covariance** of the augmented observation $\tilde{y}_\text{new}$
- $c(x', x_\text{new}) \in \mathbb{R}^{1+d}$ — **posterior cross-covariance** between
  $f(x')$ and the augmented observation:
  $$c_0 = \text{Cov}_n(f(x'),\, f(x_\text{new})), \qquad c_l = \text{Cov}_n(f(x'),\, \partial f/\partial x_l(x_\text{new}))$$
- $L_\text{new}$ — Cholesky factor: $\Sigma_n(x_\text{new}) = L_\text{new} L_\text{new}^\top$
- $b(x', x_\text{new}) = L_\text{new}^{-1}\, c(x', x_\text{new}) \in \mathbb{R}^{1+d}$ — **influence vector**

### 3.2 Mean update (stochastic)

The innovation at $x_\text{new}$ is $\tilde{y}_\text{new} - \tilde{\mu}_n(x_\text{new}) = L_\text{new}\varepsilon$,
$\varepsilon \sim \mathcal{N}(0, I_{1+d})$.  The posterior mean at any $x'$ updates as:

$$\boxed{\mu_{n+1}(x') = \mu_n(x') + b(x', x_\text{new})^\top \varepsilon}$$

The update is **stochastic** (depends on what $\tilde{y}_\text{new}$ is observed).

### 3.3 Variance update (deterministic)

The posterior variance reduces **deterministically** — independent of the observed value:

$$\boxed{\sigma_{n+1}^2(x') = \sigma_n^2(x') - \|b(x', x_\text{new})\|^2}$$

This determinism is a key property of Gaussian posteriors.  Plain-GP acquisitions
(IMIQR, EIV) that integrate over hypothetical $y_\text{new}$ values become
**exact and MC-free** for derivative GPs because the variance reduction is fixed.

### 3.4 Comparison with plain GP

For a plain GP observing only $f(x_\text{new})$, the influence is scalar:

$$b_\text{plain}(x') = \frac{\text{Cov}_n(f(x'), f(x_\text{new}))}{\sigma_n(x_\text{new})}, \qquad \sigma_{n+1}^2(x') = \sigma_n^2(x') - b_\text{plain}^2$$

The derivative GP replaces this scalar with the richer vector $b \in \mathbb{R}^{1+d}$,
giving $\|b\|^2 \geq b_\text{plain}^2$ — strictly larger variance reduction whenever
gradient observations carry additional information (always true for smooth kernels).

---

## 4. Category I — Point-Based Acquisitions

These acquisitions select $x_\text{new}$ based only on the **current** GP state at that
point, without modelling what happens after the observation.

### 4.1 MaxVar

$$\text{MaxVar}(x) = \sigma_n^2(x)$$

### 4.2 LogMaxVar

$$\text{LogMaxVar}(x) = \log \sigma_n^2(x) + \log q_n(x)$$

(log-posterior-weighted variance in likelihood space)

### 4.3 MWMV (Mass-Weighted Mean Variance)

$$\text{MWMV}(x) = \frac{1}{S} \sum_i w_i\, \sigma_n^{(i)\,2}(x)$$

where $w_i$ are normalised evidence weights over sensor subsets.

### 4.4 Effect of gradient GP on point-based acquisitions

**No formula change is needed.**  The gradient GP automatically provides a more
accurate posterior (smaller $\sigma_n^2$, better $\mu_n$) because each evaluation
contributes $1+d$ virtual observations instead of $1$.  The acquisition landscapes
are sharper and converge faster, but the selection criterion is identical.

---

## 5. Category II — Lookahead Acquisitions

These acquisitions estimate the expected improvement in inference quality from adding
a new observation at $x_\text{new}$.

### 5.1 General structure

All lookahead acquisitions integrate a **quality metric** $Q$ over the domain,
comparing the one-step posterior to the current one:

$$\text{Acq}(x_\text{new}) = \int_\mathcal{X} \bigl[Q_n(x') - \mathbb{E}\!\bigl[Q_{n+1}(x')\bigr]\bigr] \cdot q_n(x') \, dx'$$

or its MC approximation over a fantasy grid $\{x'_1, \ldots, x'_M\}$ with posterior
weights $\tilde{w}_m \propto q_n(x'_m)$.

The choice of quality metric $Q$ and whether it involves the stochastic mean or only
the deterministic variance determines the acquisition's character:

| Metric $Q$ | Depends on mean? | MC over $\varepsilon$ needed? | Acquisition |
|------------|-----------------|-------------------------------|-------------|
| $\sigma_n^2(x')$ | No | **No** — variance update is deterministic | dIVR / dEIV |
| $\text{IQR}_n(x')$ | No | **No** — IQR depends only on $\sigma^2$ | dIMIQR |
| $\|\mu_{n+1}(x') - \mu_n(x')\|$ | Yes | Yes — mean update is stochastic | dIMMD |
| $\max_{x'} \ell(\mu_{n+1}(x'), \sigma^2)$ | Yes | Yes | dKG |

---

## 6. Derivative-GP Derivations

### 6.1 dIVR — Integrated Variance Reduction *(= derivative-GP EIV)*

**Plain-GP EIV** minimises the expected integrated posterior variance by sampling
$y_\text{new} \sim \mathcal{N}(\mu_n(x_\text{new}), \sigma_n^2(x_\text{new}))$, adding it to the
dataset, and averaging the result.  Each sample requires refitting the GP — expensive.

With a derivative GP, the variance update is **deterministic** (§3.3), so the
expectation over $y_\text{new}$ is trivial:

$$\mathbb{E}_{y_\text{new}}\!\bigl[\sigma_{n+1}^2(x')\bigr] = \sigma_n^2(x') - \|b(x', x_\text{new})\|^2$$

The expected integrated variance after observation is therefore:

$$\mathbb{E}\!\left[\int q_n(x') \sigma_{n+1}^2(x') \, dx'\right]
  \approx \sum_m \tilde{w}_m \bigl(\sigma_n^2(x'_m) - \|b_m\|^2\bigr)$$

Maximising the **reduction** (i.e. maximising how much variance is removed) gives:

$$\boxed{\text{dIVR}(x_\text{new}) = \sum_m \tilde{w}_m\, \|b_m\|^2}$$

No sampling of $y_\text{new}$ is needed.  See [divr-bosip.md](divr-bosip.md) for the
full derivation.

---

### 6.2 dIMIQR — Integrated Median IQR

**Plain-GP IMIQR** fixes the speculative observation at the median predictive value
(i.e. $y_\text{new} = \mu_n(x_\text{new})$, so $\varepsilon = 0$) and integrates the
IQR of the resulting posterior log-likelihood over the domain.

The IQR of $p(z_o \mid f(x'))$ under $f(x') \mid \mathcal{D}_{n+1} \sim \mathcal{N}(\mu_n(x'), \sigma_{n+1}^2(x'))$
depends only on the **variance** $\sigma_{n+1}^2$, not on the specific value observed.
For the two likelihood types:

**Normal likelihood** — $\ell(y) = \log \mathcal{N}(z_o; y, \sigma_\text{obs}^2)$:

$$\text{IQR}_n(x') = 2\Phi^{-1}(0.75) \cdot \sqrt{\sigma_n^2(x') + \sigma_\text{obs}^2}$$

After the derivative-GP update $\sigma_{n+1}^2 = \sigma_n^2 - \|b_m\|^2$:

$$\text{IQR}_{n+1}(x'_m) = 2\Phi^{-1}(0.75) \cdot \sqrt{\sigma_n^2(x'_m) - \|b_m\|^2 + \sigma_\text{obs}^2}$$

**Exponential likelihood** — GP models $\log p(z_o \mid x)$ directly:

$$\text{IQR}_{n+1}(x'_m) = 2\sinh\!\bigl(u \cdot \sqrt{\sigma_n^2(x'_m) - \|b_m\|^2}\bigr), \qquad u = \Phi^{-1}(0.75)$$

In both cases the dIMIQR acquisition is **fully deterministic**:

$$\boxed{\text{dIMIQR}(x_\text{new}) = -\sum_m \tilde{w}_m \cdot \text{IQR}_{n+1}(x'_m, x_\text{new})}$$

The sign is negative because we maximise the acquisition (minimise the IQR after the
observation).

---

### 6.3 dIMMD — Integrated Mean Maximum Discrepancy

**Plain-GP IMMD** measures the expected absolute mean shift, integrated over the
posterior, as a proxy for information gain.  With a derivative GP, the mean update at
$x'_m$ is $b_m^\top \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I_{1+d})$:

$$b_m^\top \varepsilon \sim \mathcal{N}(0,\, \|b_m\|^2)$$

The expected absolute mean shift is the half-normal mean:

$$\mathbb{E}_\varepsilon\!\bigl[|b_m^\top \varepsilon|\bigr] = \|b_m\| \cdot \sqrt{2/\pi}$$

Integrating over the posterior:

$$\boxed{\text{dIMMD}(x_\text{new}) = \sqrt{\frac{2}{\pi}}\sum_m \tilde{w}_m\, \|b_m\|}$$

The constant $\sqrt{2/\pi}$ is irrelevant for maximisation.  This acquisition is also
**deterministic** despite involving the mean update, because only the expected
magnitude (norm of $b_m$) matters, not the direction.

---

### 6.4 dKG — Derivative Knowledge Gradient

**dKG** estimates the expected improvement in the best achievable expected
log-posterior over a fantasy grid:

$$\text{dKG}(x_\text{new})
  = \mathbb{E}_\varepsilon\!\left[\max_m \bigl(\ell(\mu_n(x'_m) + b_m^\top\varepsilon,\; \sigma_n^2(x'_m) - \|b_m\|^2) + \log p(x'_m)\bigr)\right]
  - \max_m \bigl(\ell(\mu_n(x'_m),\; \sigma_n^2(x'_m)) + \log p(x'_m)\bigr)$$

where $\ell(\mu, \sigma^2) = \mathbb{E}_{y \sim \mathcal{N}(\mu,\sigma^2)}[\log p(z_o \mid y)]$.

Unlike dIVR/dIMMD/dIMIQR, dKG requires **Monte Carlo over $\varepsilon$** because the
$\max_m$ is a nonlinear function of the stochastic mean update.  See the full
derivation in §8 below and the note on mode-seeking behaviour in §7.

---

## 7. Comparison Table

| Acquisition | Objective | Mean update? | MC? | Mode-seeking? | GP variance role |
|-------------|-----------|:---:|:---:|:---:|---|
| MaxVar | $\max_x \sigma_n^2(x)$ | — | No | No | Direct |
| LogMaxVar | $\max_x \log\sigma_n^2(x) + \log q_n(x)$ | — | No | Mild | Direct |
| MWMV | Weighted $\sigma_n^2$ across sensor subsets | — | No | No | Direct |
| **dIVR** | $\sum_m \tilde{w}_m \|b_m\|^2$ | No | **No** | No | Via $\|b_m\|^2$ |
| **dIMIQR** | $-\sum_m \tilde{w}_m \text{IQR}_{n+1}(x'_m)$ | No | **No** | No | Via $\sigma_{n+1}^2$ |
| **dIMMD** | $\sum_m \tilde{w}_m \|b_m\|$ | Yes (expected) | **No** | No | Via $\|b_m\|$ |
| **dKG** | $\mathbb{E}[\max_m \ell(\mu_{n+1}(x'_m),\sigma^2_{n+1}(x'_m)) + \log p]$ | Yes | Yes | **Yes** | Via $\|b_m\|^2$ |

**Notes:**

- *Mode-seeking*: dKG concentrates sampling at the posterior mode because the $\max_m$
  is always dominated by the fantasy point nearest to $z_o$.  The sum-based acquisitions
  (dIVR, dIMMD, dIMIQR) avoid this by weighting all posterior-relevant points equally.

- *Relationship between dIVR, dIMMD, dIMIQR*: all three are deterministic and use the
  same influence vectors $b_m$.  They differ in how variance reduction $\|b_m\|^2$ is
  translated into an inference improvement score: $\|b_m\|^2$ (variance), $\|b_m\|$
  (standard deviation / mean absolute shift), $\text{IQR}_{n+1}$ (quantile range).

- *dIVR = derivative-GP EIV*: the original EIV samples $y_\text{new}$ and refits the GP.
  For derivative GPs this is unnecessary because the variance reduction is deterministic;
  dIVR is the exact closed-form version.

- *Gradient GP advantage over plain GP*: all acquisitions benefit from larger $\|b_m\|^2$
  per evaluation (gradient information) and from faster posterior convergence.

---

## 8. Full dKG Derivation

*(This section retains the original detailed derivation for reference.)*

### 8.1 One-step posterior mean update

After adding the augmented observation $\tilde{y}_\text{new}$ at $x_\text{new}$:

$$\mu_{n+1}(x') = \mu_n(x') + b(x', x_\text{new})^\top \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I_{1+d})$$

### 8.2 Likelihood-aware objective

For BOSIP the quantity of interest is the expected log-likelihood:

$$\ell(\mu, \sigma^2) = \mathbb{E}_{y \sim \mathcal{N}(\mu,\sigma^2)}\!\bigl[\log p(z_o \mid y)\bigr]$$

**Normal likelihood** (Gaussian convolution):

$$\ell(\mu, \sigma^2) = \log \mathcal{N}(z_o;\; \mu,\; \sigma^2 + \sigma_\text{obs}^2)$$

**Exponential likelihood** (GP models $\log p$ directly):

$$\ell(\mu, \sigma^2) = \mu + \tfrac{\sigma^2}{2}$$

### 8.3 BOSIP-dKG formula

$$\boxed{
\text{dKG}(x_\text{new})
  = \frac{1}{S}\sum_{s=1}^S
    \max_m\!\Bigl[\ell\!\bigl(\mu_n(x'_m) + b_m^\top\varepsilon_s,\; \sigma_n^2(x'_m) - \|b_m\|^2\bigr)
                  + \log p(x'_m)\Bigr]
  - \max_m\!\Bigl[\ell\!\bigl(\mu_n(x'_m),\; \sigma_n^2(x'_m)\bigr) + \log p(x'_m)\Bigr]
}$$

### 8.4 Why dKG is mode-seeking (Normal likelihood)

For concave likelihoods, the expected mean update and variance reduction **cancel in
expectation**:

$$\mathbb{E}_\varepsilon\!\bigl[\ell(\mu + b_m^\top\varepsilon,\; \sigma^2 - \|b_m\|^2)\bigr]
= \ell(\mu,\; \sigma^2)$$

So dKG gain comes entirely from the variance of $\max_m$.  The $\max$ is dominated by
the fantasy point $m^*$ nearest to $z_o$ (highest $\ell$ value).  To maximally influence
$m^*$, $x_\text{new}$ is placed near it — which is the posterior mode.  Sum-based
acquisitions (dIVR, dIMMD) avoid this because they use $\Sigma_m$ instead of $\max_m$.

---

## 9. References

- **Wu, J., Poloczek, M., Wilson, A. G., & Frazier, P. I. (2017).**
  *Bayesian Optimization with Gradients.*
  NeurIPS 30.

- **Scott, W., Frazier, P. I., & Powell, W. B. (2011).**
  *The Correlated Knowledge Gradient for Simulation Optimization of Continuous
  Parameters using Gaussian Process Regression.*
  SIAM Journal on Optimization, 21(3).

- **Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989).**
  *Design and Analysis of Computer Experiments.*
  Statistical Science, 4(4).

- **Järvenpää, M., Gutmann, M. U., Vehtari, A., & Marttinen, P. (2021).**
  *Parallel Gaussian Process Surrogate Bayesian Inference with Noisy Likelihood
  Evaluations.*
  Bayesian Analysis, 16(1).
