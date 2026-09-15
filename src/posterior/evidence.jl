
"""
    evidence(post, proposal; kwargs...)
    evidence(post, proposal, domain_area; kwargs...)

Return the estimated normalization constant (evidence) ``\\hat{p}(z_o)``
of the unnormalized pdf provided as `post`.

Non-log wrapper around [`log_evidence`](@ref).

## Arguments
- `post`: A function `::AbstractVector{<:Real} -> ::Real`
        representing any unnormalized pdf, typically the unnormalized posterior ``\\hat{p}(z_o|x) p(x)``.
- `proposal`: A multivariate distribution used to draw parameter samples.
        Typically the prior ``p(x)``.
- `domain_area`: The area of the parameter domain. Only required for the
        self-normalized estimator. See [`log_evidence`](@ref).

## Keywords
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `proposal` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `isnothing(xs)`.

## See Also

[`log_evidence`](@ref)
"""
function evidence(post::Base.Callable, proposal::MultivariateDistribution, args...; kwargs...)
    logpost(x) = log(post(x))
    log_ev = log_evidence(logpost, proposal, args...; kwargs...)
    return exp(log_ev)
end

"""
    log_evidence(logpost, proposal; kwargs...)
    log_evidence(logpost, proposal, domain_area; kwargs...)

Return the log of the estimated normalization constant (evidence) ``\\log \\hat{p}(z_o)``
of the unnormalized pdf provided as `logpost`.

The first method uses a standard importance-sampling Monte Carlo estimator:

``\\hat{p}(z_o) = \\frac{1}{N} \\sum_i \\frac{f(x_i)}{q(x_i)}, \\quad x_i \\sim q``

The second method uses a self-normalized importance-sampling Monte Carlo estimator:

``\\hat{p}(z_o) = S \\cdot \\frac{\\sum_i w_i f(x_i)}{\\sum_i w_i}, \\quad w_i = \\frac{1}{q(x_i)}, \\quad x_i \\sim q``

where ``f(x)`` is the unnormalized pdf and ``S`` is the domain area.
The self-normalized variant can reduce variance when the proposal ``q`` does not exactly match the prior ``p(x)``.

## Arguments
- `logpost`: A function `::AbstractVector{<:Real} -> ::Real`
        representing the log of any unnormalized pdf, typically the unnormalized posterior ``\\log \\hat{p}(z_o|x) p(x)``.
- `proposal`: A multivariate distribution ``q`` used to draw parameter samples.
        Typically the prior ``p(x)``, in which case the importance weights cancel.
- `domain_area`: The area (volume) ``S`` of the parameter domain.
        Only required for the self-normalized estimator.

## Keywords
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `proposal` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `isnothing(xs)`.

## See Also

[`evidence`](@ref)
"""
function log_evidence(logpost::Base.Callable, proposal::MultivariateDistribution;
    xs = nothing,
    samples = 10_000,
)
    isnothing(xs) && (xs = rand(proposal, samples))
    N = size(xs, 2)

    log_ws = 0. - logpdf.(Ref(proposal), eachcol(xs))
    log_vals = logpost.(eachcol(xs))

    # ev = (1 / N) * sum(ws .* vals)
    log_ev = logsumexp(log_ws .+ log_vals) - log(N)
    return log_ev
end
function log_evidence(logpost::Base.Callable, proposal::MultivariateDistribution, domain_area::Real;
    xs = nothing,
    samples = 10_000,
)
    isnothing(xs) && (xs = rand(proposal, samples))
    S = domain_area

    log_ws = 0. - logpdf.(Ref(proposal), eachcol(xs))
    log_vals = logpost.(eachcol(xs))

    # ev = (S / sum(ws)) * sum(ws .* vals)
    log_ev = log(S) - logsumexp(log_ws) + logsumexp(log_ws .+ log_vals)
    return log_ev
end
