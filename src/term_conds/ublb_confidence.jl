
"""
    UBLBConfidence(; kwargs...)

Calculates the `q`-confidence region of the UB and LB approximate posterior.
Terminates after the IoU of the two confidence intervals surpasses `r`.
The UB and LB confidence intervals are calculated using the GP mean +- `n` GP stds.

# Keywords

- `max_iters::Union{Nothing, <:Int}`: The maximum number of iterations.
- `samples::Int`: The number of samples used to approximate the confidence regions
        and their IoU ratio. Only has an effect if `isnothing(xs)`.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of parameter samples from the `x_prior` defined in `BosipProblem`.
- `n::Float64`: The number of predictive deviations added/substracted from the GP mean
        to get the two posterior approximations. Defaults to `n = 1.`.
- `q::Float64`: The confidence value of the confidence regions.
        Defaults to `q = 0.8`.
- `r::Float64`: The algorithm terminates once the IoU ratio surpasses `r`.
        Defaults to `r = 0.8`.
"""
struct UBLBConfidence{
    I<:Union{IterLimit, NoLimit},
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BosipTermCond
    iter_limit::I
    samples::Int
    xs::X
    n::Float64
    q::Float64
    r::Float64
end
function UBLBConfidence(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    n = 1.,
    q = 0.8,
    r = 0.8,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return UBLBConfidence(iter_limit, samples, xs, n, q, r)
end

function (cond::UBLBConfidence)(bosip::BosipProblem)
    @assert bosip.problem.params isa UniFittedParams
    return ublb_confidence(cond, bosip)
end

function ublb_confidence(cond::UBLBConfidence, bosip::BosipProblem{Nothing})
    cond.iter_limit(bosip.problem) || return false
    ratio = calculate(cond, bosip) 
    return ratio < cond.r
end

function ublb_confidence(cond::UBLBConfidence, bosip::BosipProblem{Matrix{Bool}})
    cond.iter_limit(bosip.problem) || return false
    (bosip.problem.data isa ExperimentData) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bosip), eachcol(bosip.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::UBLBConfidence, bosip::BosipProblem)
    if isnothing(cond.xs)
        xs = rand(bosip.x_prior, cond.samples)
    else
        xs = cond.xs
    end

    gp_post = BOSS.model_posterior(bosip.problem)
    gp_lb = gp_bound(gp_post, -cond.n)
    gp_ub = gp_bound(gp_post, +cond.n)

    like_lb = approx_likelihood(bosip.likelihood, bosip, gp_lb)
    like_ub = approx_likelihood(bosip.likelihood, bosip, gp_ub)

    x_prior = bosip.x_prior
    f_lb(x) = pdf(x_prior, x) * like_lb(x)
    f_ub(x) = pdf(x_prior, x) * like_ub(x)

    xs_logpdf = logpdf.(Ref(x_prior), eachcol(xs))
    ws_lb = exp.(  log.(f_lb.(eachcol(xs))) .- xs_logpdf )
    ws_ub = exp.(  log.(f_ub.(eachcol(xs))) .- xs_logpdf )

    c_lb = find_cutoff(f_lb, xs, ws_lb, cond.q)
    c_ub = find_cutoff(f_ub, xs, ws_ub, cond.q)

    in_lb = (f_lb.(eachcol(xs)) .> c_lb)
    in_ub = (f_ub.(eachcol(xs)) .> c_ub)
    return set_iou(in_lb, in_ub, bosip.x_prior, xs)
end
