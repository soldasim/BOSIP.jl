
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
        set of parameter samples from the `x_prior` defined in `BolfiProblem`.
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
} <: BolfiTermCond
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

function (cond::UBLBConfidence)(bolfi::BolfiProblem{<:Any, Nothing})
    cond.iter_limit(bolfi.problem) || return false
    ratio = calculate(cond, bolfi) 
    return ratio < cond.r
end

function (cond::UBLBConfidence)(bolfi::BolfiProblem{<:Any, Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::UBLBConfidence, bolfi::BolfiProblem)
    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end

    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_lb = gp_bound(gp_post, -cond.n)
    gp_ub = gp_bound(gp_post, +cond.n)

    f_lb = approx_posterior(gp_lb, bolfi.x_prior, std_obs(bolfi); xs)
    f_ub = approx_posterior(gp_ub, bolfi.x_prior, std_obs(bolfi); xs)
    f_lb, c_lb = find_cutoff(f_lb, bolfi.x_prior, cond.q; xs)
    f_ub, c_ub = find_cutoff(f_ub, bolfi.x_prior, cond.q; xs)

    in_lb = (f_lb.(eachcol(xs)) .> c_lb)
    in_ub = (f_ub.(eachcol(xs)) .> c_ub)
    return set_iou(in_lb, in_ub, bolfi.x_prior, xs)
end
