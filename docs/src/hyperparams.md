# Hyperparameters

It is important to define reasonable values and priors for the hyperparameters. Poorly designed priors can cause the method to perform suboptimally, or cause numerical issues. This page contains some reasonable defaults for defining the hyperparameters.

## Parameter Prior

The parameter prior describes our expert knowledge about the domain. If we have limited knowledge about the parameters, the uniform prior can be used, which will not affect the optimization at all. Or one might for example use a zero-centered multivariate normal prior to suppress parameters with too large absolute values.

## Likelihood

The likelihood ``p(y_o|x)`` describes the stochastic process which generated the real experiment observation ``y_o``. In general, the likelihood somehow depends on the simulator output ``y = g(x)``. In most cases, one can view the simulator as an oracle providing the true value ``y = g(x) \approx f_t(x)`` and the likelihood describes the nature of the noise of the real observation ``y_o \sim p(y_o|f_t(x))`.

In the case of the `NormalLikelihood`, one needs to define the observation noise deviation ``\sigma_f``. This deviation has to be estimated by the user and provided as a vector-valued constant. It should reflect the measurement precision in the real experiment used to obtain the observation ``y_o``. The value of ``\sigma_f`` greatly affects the width of the resulting posterior. Thus some care should be taken with its choice.

## Kernel

The kernel is a hyperparameter of the Gaussian process. It controls how the data affect the predictions of the GP in different parts of the domain. I recommend using one of the Matérn kernels, for example the Matérn``\frac{3}{2}`` kernel. The Matérn kernels are a common choice in Bayesian optimization.

## Length Scales

The length scales control the distance withing the parameter domain, at which the data sitll affect the prediction of the GP. Given that we have ``n`` parameters and ``m`` observation dimensions, there are in total ``n \times m`` length scales. For each observation dimension ``1,...,m``, we need to define a separate ``n``-variate length scale prior.

To define a weak length scale prior, it is reasonable to use the half-normal distribution
```math
TR_0 \left[ \mathcal{N}\left( 0, \left(\frac{ub - lb}{3}\right)^2 \right) \right] \;,
```
where ``lb, ub`` are the lower and upper bounds of the domain. Such prior will suppress length scales higher than the size of the domain.

A slightly more robust option is to use the inverse gamma prior to suppress exceedingly small length scales as well. One construct such prior as
```math
\begin{aligned}
\text{Inv-Gamma}(\alpha, \beta) \\
\alpha = \frac{\mu^2}{\sigma^2} + 2 \\
\beta = \mu (\frac{\mu^2}{\sigma^2} + 1) \\
\mu = (\lambda_{max} + \lambda_{min}) / 2 \\
\sigma = (\lambda_{max} - \lambda_{min}) / 6 \;,
\end{aligned}
```
where ``\lambda_{min}, \lambda_{max}`` are the minimum and maximum allowed length scale values.

## Amplitude

The amplitude is another hyperparameter of the Gaussian process. It controls the expected degree of fluctation of the predicted values. We need to define a univariate prior for each observation dimension.

We usually do not know the exact range of function values a priori. Thus, we should be cautious with the prior. If we expect to observe values in range ``\left< y_{min}, y_{max} \right>``, a reasonable prior could a half-normal distribution
```math
TR_0 \left[ \mathcal{N}\left( 0, \left(\frac{y_{max} - y_{min}}{2}\right)^2 \right) \right] ;.
```
Such prior will prioritize amplitudes within the expected range, while still allowing slightly larger amplitudes than we expected, in case we were wrong about our assumptions.

Again, one might also construct an inverse gamma prior to additionally suppress small amplitudes. See the length scales subsection.

## Simulation Noise

We do not have to define the simulation noise deviations as exact values, as in the case of the observation noise. It is sufficient to provide priors, and BOLFI.jl will estimate the simulation noise by itself.

We can use a more or less weak prior, depending on our confidence in estimating the simulation noise. Again, a reasonable choice is to use etiher the half-normal distribution to suppress exceedingly large noise deviations, or the inverse gamma distribution to also suppress small deviations.

## Sub-Algorithms

We also need to define which algorithms should be used to estimate the model hyperparameters and maximize the acquisition function.

For simple toy experiment, I recommend using the `BOSS.SamplingMAP` model fitter, and the `BOSS.SamplingAM` or `BOSS.GridAM` acquisition maximizers.

For real problems, I recommend using the Powell's blackbox optimization algorithms from the PRIMA package. The NEWUOA algorithm for unconstrained optimization can be used for the MAP estimation of the model hyperparameters, and the BOBYQA algorithm for box-constrained optimization can be used for the acquisition maximization. To use any optimization algorithms, use the `BOSS.OptimizationMAP` model fitter and the `BOSS.OptimizationAM` acquisition maximizer.

## Termination Condition

Finally, we need to define a termination condition. A default choice would be terminating the procedure simply after a predefined number of iterations by using the `BOSS.IterLimit`.

In case one has a low-dimensional parameter domain, the `AEConfidence` and `UBLBConfidence` termination conditions can be used for an automatic convergence detection.
