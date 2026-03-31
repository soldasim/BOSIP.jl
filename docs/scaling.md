# Scaling the Gradient GP: Options and Suitability for BOSIP

This document analyses the $O(n^3 d^3)$ computational bottleneck of
`GradientGaussianProcess`, surveys available approaches to reduce it, and
discusses their suitability for BOSIP's sequential inference setting.

---

## 1. Source of the $O(n^3 d^3)$ Cost

Each of the $n$ simulator evaluations returns $1 + d$ values: one function value
and $d$ partial derivatives.  The joint GP over values and all derivatives has a
kernel matrix of size $n(1+d) \times n(1+d)$.  Cholesky factorisation of this
matrix costs:

$$O\!\bigl(n^3(1+d)^3\bigr) \approx O(n^3 d^3)$$

For BOSIP, $n$ is deliberately kept small — that is the purpose of active
learning.  The dominant bottleneck is therefore the **$d^3$ factor**, not $n^3$.
Any practical scaling strategy must target $d$.

---

## 2. Available Options

### 2.1 Partial Gradient Observations

Instead of observing all $d$ partial derivatives, observe only $k \ll d$
directional derivatives:

$$\tilde{y} = \bigl[f(x),\; v_1^\top \nabla f(x),\; \ldots,\; v_k^\top \nabla f(x)\bigr]^\top \in \mathbb{R}^{1+k}$$

The directions $v_i \in \mathbb{R}^d$ can be chosen by the acquisition function,
by active subspace analysis, or uniformly at random.  Kernel matrix size becomes
$n(1+k) \times n(1+k)$ and Cholesky costs $O(n^3 k^3)$.

**Suitability: excellent.** This is the most natural reduction strategy for
BOSIP.  If the posterior concentrates in a low-dimensional subspace of $\mathcal{X}$
— which it typically does as $n$ grows — then $k = 2$–$5$ directions capture most
of the information.  The acquisition functions dKG, dIVR, and dIMMD can be
extended to jointly optimise the query location $x_\text{new}$ and the
observation directions $\{v_i\}$.

---

### 2.2 Active Subspace Reduction

Approximate $f(x) \approx g(W^\top x)$ where $W \in \mathbb{R}^{d \times k}$,
$k \ll d$.  Fit a plain or gradient GP in $\mathbb{R}^k$ after projecting.  The
projection matrix $W$ can be estimated from gradient observations already in
$\mathcal{D}_n$ as the leading $k$ eigenvectors of the empirical matrix:

$$\hat{C} = \frac{1}{n}\sum_{i=1}^n \nabla f(x_i)\, \nabla f(x_i)^\top \in \mathbb{R}^{d \times d}$$

After reduction, cost is $O(n^3 k^3)$ with $O(n d)$ overhead for projection.

**Suitability: good but conditional.** Works well when $f$ is genuinely
low-dimensional.  Fails silently if the active subspace assumption is violated,
which may go undetected because the GP posterior appears confident.  The gradient
observations already gathered by BOSIP make estimating $W$ cheap — a natural
synergy.  Best used as an exploratory diagnostic rather than a hard compression.

---

### 2.3 MVM-Based Inference (Conjugate Gradients + KeOps)

Avoid forming and factorising the full kernel matrix.  Replace Cholesky with
iterative conjugate gradients (CG) using custom matrix-vector products (MVMs).
Each MVM evaluates $K v$ without materialising the $n(1+d) \times n(1+d)$ matrix.
Memory: $O(nd)$.  Cost per CG iteration: $O(n^2 d^2)$ (see §3 for the structural
improvement to $O(n^2 d)$).  Total with $t$ CG steps: $O(t n^2 d^2)$.

**Suitability: good for moderate $n$ and $d$.** GPyTorch's blackbox-MVM framework
(Gardner et al., 2018) provides this machinery.  It does not change the
asymptotic scaling but removes the $O(n^2 d^2)$ memory footprint and is
GPU-accelerable.  For BOSIP with $n \leq 200$ and $d \leq 20$ this is likely
sufficient without further approximation.

---

### 2.4 Random Fourier Features for Derivative Kernels

For stationary kernels, $k(x,x') = \mathbb{E}_\omega[\phi_\omega(x)\phi_\omega(x')]$
with $\phi_\omega(x) = \cos(\omega^\top x + b)$.  The gradient feature is:

$$\nabla_x \phi_\omega(x) = -\omega\,\sin(\omega^\top x + b)$$

With $m$ sampled frequencies the full augmented feature matrix has $m(1+d)$
columns.  Inference costs $O(n m^2 d^2)$, avoiding the $n^3$ term entirely.

**Suitability: promising but underexplored.** This directly addresses the
$d^3$ bottleneck.  A residual $d^2$ term comes from the derivative feature
vectors $\omega \otimes \sin(\cdot)$.  No published implementation for
BOSIP-style acquisitions is known.  The approximation error is controlled by $m$
but may be difficult to calibrate for the posterior quality metrics used by BOSIP.

---

### 2.5 Sparse Inducing-Point Methods (FITC / VFE / SVGP)

Replace $n$ training points with $m \ll n$ inducing points.  The kernel matrix
is compressed to $O(m \times n(1+d))$ and factorisation costs $O(m^3(1+d)^3)$.

**Suitability: poor for BOSIP.** These methods address the $n^3$ bottleneck, not
$d^3$.  Since BOSIP is specifically designed to keep $n$ small, this is the wrong
axis.  Inducing-point approximations also degrade posterior uncertainty
quantification — critical for all lookahead acquisition functions in BOSIP.

---

### 2.6 Additive / ANOVA Kernels

Assume $f(x) = \sum_i f_i(x_i)$ or $\sum_{i < j} f_{ij}(x_i, x_j)$.  The
kernel factorises; gradient blocks are block-diagonal.  Cost reduces to
$O(n^3 d)$ for order-1 and $O(n^3 d^2)$ for order-2 additive GPs.

**Suitability: limited for BOSIP.** Additivity is a strong structural assumption
rarely satisfied by physical simulators.  Multi-modal posteriors in correlated
directions would be misrepresented.  May be useful as a fast baseline or sanity
check.

---

### 2.7 Kronecker / Grid Structure

When inputs lie on a Cartesian product grid, the gradient kernel factorises as a
Kronecker product of 1-D kernels and their derivatives.  Cost: $O(n^{3/d} d^3)$
per dimension with $O(n \log n)$ MVMs.

**Suitability: not applicable to BOSIP.** Sequential active learning does not
produce grid-structured inputs and cannot be forced onto a grid without
reinterpreting the acquisition problem.

---

## 3. Structural Implications of the Gradient GP Kernel

The structure of the gradient kernel has direct implications for efficient
implementation that are independent of the approximation strategies above.

### 3.1 Hadamard factorisation of derivative blocks

For any **stationary kernel** $k(r)$, $r = x - x'$, the cross-derivative kernel
entry is:

$$k_{\partial_i \partial_j}(x, x') = -\partial_{r_i}\partial_{r_j}\, k(r)$$

For the **RBF kernel** with lengthscale $\ell$:

$$k_{\partial_i \partial_j}(x,x') = \left(\frac{\delta_{ij}}{\ell^2} -
\frac{(x_i - x'_i)(x_j - x'_j)}{\ell^4}\right) k_\text{RBF}(x,x')$$

The $n \times n$ matrix for the $(i,j)$ derivative block is therefore:

$$[K_{\partial_i \partial_j}]_{ab} = \left(\frac{\delta_{ij}}{\ell^2}\, [K_{ff}]_{ab}
- \frac{1}{\ell^4}\, \Delta_i \Delta_j^\top \odot K_{ff}\right)$$

where $\Delta_i \in \mathbb{R}^n$ is the vector of $i$-th coordinate differences
$(\Delta_i)_a = x_{a,i} - x_{b,i}$ along the relevant row.  Every derivative
block is a **Hadamard product** of the base kernel matrix with a rank-1 +
diagonal matrix.

For **Matérn-$\nu$** the structure is analogous: the second derivative introduces
additional polynomial terms in $\|r\|$, all expressible as Hadamard products
against $K_{ff}$ with rank-2 corrections.

### 3.2 Consequences for storage and computation

| Quantity | Naive | Using Hadamard structure |
|----------|-------|--------------------------|
| Storage | $O(n^2 d^2)$ | $O(n^2 + nd)$ |
| MVM cost | $O(n^2 d^2)$ | $O(n^2 d)$ |
| CG total ($t$ iters) | $O(t n^2 d^2)$ | $O(t n^2 d)$ |

The full $n(1+d) \times n(1+d)$ matrix is determined entirely by $K_{ff}$
($n^2$ scalars) and the $d$ coordinate difference vectors ($n^2 d$ scalars in
total).  It need not be materialised.

For a MVM $K_\text{grad}\, v$ where $v \in \mathbb{R}^{n(1+d)}$, the Hadamard
structure allows evaluating each $n$-dimensional output block using $K_{ff}$
once (applied to one slice of $v$) plus $d$ rank-1 corrections — $O(n^2)$
per correction, $O(n^2 d)$ total across all blocks.

### 3.3 Implication for acquisition function evaluation

Each acquisition function evaluation (dKG, dIVR, dIMMD) requires the predictive
covariance $\Sigma_n(x_\text{new}) \in \mathbb{R}^{(1+d)\times(1+d)}$ and the
cross-covariance vector $c(x', x_\text{new}) \in \mathbb{R}^{1+d}$ (see
[dgp-bosip.md](dgp-bosip.md) §3).  These involve solving $K_\text{grad}^{-1}$
applied to $(1+d)$ right-hand sides.  With a CG solver using Hadamard MVMs this
costs $O(t n^2 d)$ per acquisition evaluation rather than $O(n^3 d^3)$ for a
full refactorisation.

---

## 4. Summary and Recommendations

| Method | Targets | Scaling | BOSIP suitability | Risk |
|--------|---------|---------|-------------------|------|
| Partial gradients ($k$ directions) | $d^3 \to k^3$ | $O(n^3 k^3)$ | **Excellent** | information loss if $k$ too small |
| Active subspace | effective $d \to k$ | $O(n^3 k^3)$ | Good | silent failure if assumption violated |
| MVM / CG (KeOps) | memory + constant | $O(t n^2 d^2)$ | Good | not a scaling improvement |
| Hadamard kernel structure | $d^2 \to d$ in MVMs | $O(t n^2 d)$ | **Excellent** | implementation effort only |
| Random Fourier features | $n^3 d^3 \to nm^2 d^2$ | $O(nm^2 d^2)$ | Promising | approximation error |
| Sparse inducing points | $n^3$ only | $O(m^3 d^3)$ | Poor | wrong bottleneck |
| Additive kernels | $d^3 \to d$ | $O(n^3 d)$ | Limited | strong structural assumption |
| Kronecker grid | both | $O(n^{3/d})$ | Not applicable | requires grid inputs |

**Near-term recommendation.** For BOSIP with $n \leq 200$ and $d \leq 20$:
exploit the Hadamard kernel structure (§3) within a CG solver to achieve
$O(n^2 d)$ MVMs.  This requires no approximation and is compatible with the
sequential design loop.

**For high $d$ ($d > 20$).** Combine MVM-based inference with partial gradient
observations (§2.1): choose $k \sim 5$ directions that maximally reduce posterior
variance (i.e., the directions along which the current acquisition has highest
sensitivity), and observe only those directional derivatives at each call.  This
reduces the per-call kernel update to $O(n^2 k)$ and acquisition evaluation to
$O(t n^2 k)$, while retaining the BOSIP posterior structure.

---

## 5. References

- **Gardner, J., Pleiss, G., Bindel, D., Weinberger, K., & Wilson, A. (2018).**
  *GPyTorch: Blackbox Matrix-Vector Multiplication based Gaussian Process
  Inference with GPU Acceleration.* NeurIPS 31.

- **De Roos, F., Gijsberts, P., & Rottmann, A. (2021).**
  *High-Dimensional Gaussian Process Inference with Derivatives.* ICML.
  (Identifies the $O(N^3 d^3)$ cost bottleneck.)

- **Constantine, P. G. (2015).**
  *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies.*
  SIAM.

- **Rahimi, A. & Recht, B. (2007).**
  *Random Features for Large-Scale Kernel Machines.* NeurIPS 20.

- **Wu, J., Poloczek, M., Wilson, A. G., & Frazier, P. I. (2017).**
  *Bayesian Optimization with Gradients.* NeurIPS 30.
