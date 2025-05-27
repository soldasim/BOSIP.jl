
"""
    AEConfidence(; kwargs...)

Calculates the `q`-confidence region of the expected and the approximate posteriors.
Terminates after the IoU of the two confidence regions surpasses `r`.

# Keywords

- `max_iters::Union{Nothing, <:Int}`: The maximum number of iterations.
- `samples::Int`: The number of samples used to approximate the confidence regions
        and their IoU ratio. Only has an effect if `isnothing(xs)`.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of parameter samples from the `x_prior` defined in `BolfiProblem`.
- `q::Float64`: The confidence value of the confidence regions.
        Defaults to `q = 0.95`.
- `r::Float64`: The algorithm terminates once the IoU ratio surpasses `r`.
        Defaults to `r = 0.95`.
"""
struct AEConfidence{
    I<:Union{IterLimit, NoLimit},
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BolfiTermCond
    iter_limit::I
    samples::Int
    xs::X
    q::Float64
    r::Float64
end
function AEConfidence(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    q = 0.95,
    r = 0.95,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return AEConfidence(iter_limit, samples, xs, q, r)
end

function (cond::AEConfidence)(bolfi::BolfiProblem)
    @assert bolfi.problem.params isa UniFittedParams
    return ae_confidence(cond, bolfi)
end

function ae_confidence(cond::AEConfidence, bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false
    ratio = calculate(cond, bolfi)
    return ratio < cond.r
end

function ae_confidence(cond::AEConfidence, bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentData) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::AEConfidence, bolfi::BolfiProblem)
    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end

    f_approx = approx_posterior(bolfi; normalize=false, xs)
    f_expect = posterior_mean(bolfi; normalize=false, xs)

    xs_logpdf = logpdf.(Ref(bolfi.x_prior), eachcol(xs))
    ws_approx = exp.( log.(f_approx.(eachcol(xs))) .-  xs_logpdf)
    ws_expect = exp.( log.(f_expect.(eachcol(xs))) .-  xs_logpdf)

    c_approx = find_cutoff(f_approx, xs, ws_approx, cond.q)
    c_expect = find_cutoff(f_expect, xs, ws_expect, cond.q)

    in_approx = (f_approx.(eachcol(xs)) .> c_approx)
    in_expect = (f_expect.(eachcol(xs)) .> c_expect)
    return set_iou(in_approx, in_expect, bolfi.x_prior, xs)
end
