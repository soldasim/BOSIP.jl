
abstract type BolfiTermCond end

struct TermCondWrapper{
    T<:BolfiTermCond
} <: TermCond
    term_cond::T
    bolfi::BolfiProblem
end

TermCondWrapper(term_cond::TermCond, ::BolfiProblem) = term_cond

(wrap::TermCondWrapper)(::BossProblem) = wrap.term_cond(wrap.bolfi)


# - - - Confidence Intervals - - - - -

struct ConfidenceTermCond{
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BolfiTermCond
    samples::Int
    xs::X
    q::Float64
    r::Float64
end
ConfidenceTermCond(;
    samples = 10_000,
    xs = nothing,
    q = 0.95,
    r = 0.95,
) = ConfidenceTermCond(samples, xs, q, r)

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Nothing})
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratio = calculate(cond, bolfi)
    return ratio < cond.r
end

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Matrix{Bool}})
    return any(
        cond.(
            (get_subset(bolfi, set) for set in eachcol(bolfi.y_sets))
        )
    )
end

function calculate(cond::ConfidenceTermCond, bolfi::BolfiProblem)
    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_med = gp_quantile(gp_post, 0.5)

    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end
    _, _, V_mean = find_cutoff(gp_post, bolfi.x_prior, bolfi.var_e, cond.q; xs)
    _, _, V_med = find_cutoff(gp_med, bolfi.x_prior, bolfi.var_e, cond.q; xs)

    return V_med / V_mean
end
