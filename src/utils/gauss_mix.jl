
# const GaussMix = MixtureModel{Multivariate, Distributions.Continuous, <:MvNormal}

"""
Contains all hyperparameters for the function `approx_by_gauss_mix`.

# Kwargs
- `algorithm`: Optimization algorithm used to find the modes.
- `multistart::Int`: Number of optimization restarts.
- `parallel::Bool`: Controls whether the individual optimization runs
        are performed in paralell.
- `autodiff::SciMLBase.AbstractADType`: Defines the autodiff library
        used for the optimization. (Only relevant if a gradient-based
        optimizer is set as `algorithm`.)
- `cluster_ϵs::Union{Nothing, Vector{Float64}}`: The minimum distance between modes. Modes which
        are too close to a "more important" mode are discarded.
        Also defines the minimum distance of a mode from a domain boundary.
- `rel_min_weight::Float64`: The minimum pdf value of a mode to be considered
        relative to the highest pdf value among all found modes.
- `kwargs...`: Other kwargs are passed to the optimization algorithm.
"""
struct GaussMixOptions{
    A<:Any,
}
    algorithm::A
    multistart::Int
    parallel::Bool
    autodiff::SciMLBase.AbstractADType
    cluster_ϵs::Union{Nothing, Vector{Float64}}
    cons_atol::Real
    rel_min_weight::Float64
    kwargs::Base.Pairs{Symbol, <:Any}
end
GaussMixOptions(;
    algorithm,
    multistart = 200,
    parallel = true,
    autodiff = AutoForwardDiff(),
    cluster_ϵs = nothing,
    cons_atol = 0.05,
    rel_min_weight = 1e-8,
    kwargs...
) = GaussMixOptions(algorithm, multistart, parallel, autodiff, cluster_ϵs, cons_atol, rel_min_weight, kwargs)

cluster_ϵs(opt::GaussMixOptions, domain::Domain) =
    isnothing(opt.cluster_ϵs) ? _default_cluster_ϵs(domain) : opt.cluster_ϵs

_default_cluster_ϵs(domain::Domain) = (domain.bounds[2] .- domain.bounds[1]) ./ (ℯ^3)

"""
Approximate the given posterior by a Gaussian mixture.

Find all modes via Optimization.jl, then approximate each mode with a mutlivariate Gaussian
with mean in the mode and variance according to the second derivation of the true posterior in the mode.
"""
function approx_by_gauss_mix(logpost, domain::Domain, opt::GaussMixOptions)
    μs = opt_for_modes(logpost, domain, opt)

    gausses = laplace_approx.(Ref(logpost), μs, Ref(domain.bounds))
    gausses = [filter(!isnothing, gausses)...]
    if isempty(gausses)
        @error "All Laplace approximations failed! (of all found modes)"
        @assert false
    end
    weights = exp.(logpost.(getfield.(gausses, Ref(:μ))))

    if isempty(gausses)
        @warn "Approximation by a Gaussian mixture failed! No extrema found in the domain."
        return nothing
    else
        return MixtureModel(gausses, weights ./ sum(weights))
    end
end

function opt_for_modes(logpost, domain::Domain, opt::GaussMixOptions)
    starts = BOSS.get_starts(opt.multistart, domain)

    # objective function
    obj_func = logpost
    cons_func = isnothing(domain.cons) ? nothing : (res, x, p) -> (res .= domain.cons(x))
    objective = OptimizationFunction((x, _) -> -obj_func(x), opt.autodiff; cons=cons_func)

    # constraints
    lb, ub = domain.bounds
    c_dim = BOSS.cons_dim(domain)
    int = domain.discrete

    # optimization problem
    function problem(start)
        return OptimizationProblem(objective, start, nothing;
            lb,
            ub,
            lcons = fill(0., c_dim),
            ucons = fill(Inf, c_dim),
            int,
        )
    end

    # optimization run
    function optimize_(start)
        x = Optimization.solve(problem(start), opt.algorithm; opt.kwargs...).u
        val = obj_func(x)  # correct sign
        return x, val
    end

    # --- MAIN ----
    θs, vals = BOSS.optimize_multistart(optimize_, starts, opt.parallel, BossOptions(); return_all=true)
    θs = cluster_modes(θs, vals; opt.rel_min_weight, cluster_ϵs=cluster_ϵs(opt, domain))
    # θs = skip_boundaries(θs, domain; ϵs=cluster_ϵs(opt, domain), opt.cons_atol) # modes at boundaries are not true extrema
    return θs
end

function cluster_modes(θs::AbstractVector{<:AbstractVector{<:Real}}, vals::AbstractVector{<:Real}; rel_min_weight::Real, cluster_ϵs::AbstractVector{<:Real})
    # sort by descending posterior value
    score = sortperm(vals; rev=true)
    θs = θs[score]
    vals = vals[score]

    # keep only a single sample from each mode
    keep = [first(θs)]
    v1 = first(vals)

    for (θ, v) in zip(θs, vals)
        (v / v1 < rel_min_weight) && continue
        is_new_mode(θ, keep, cluster_ϵs) && push!(keep, θ)
    end

    return keep
end

function skip_boundaries(θs::AbstractVector{<:AbstractVector{<:Real}}, domain::Domain; ϵs::AbstractVector{<:Real}, cons_atol::Real)
    return filter(θ -> !on_boundary(θ, domain; ϵs, cons_atol), θs)
end
function on_boundary(θ::AbstractVector{<:Real}, domain::Domain; ϵs::AbstractVector{<:Real}, cons_atol::Real)
    near(θi, b, ϵ) = isapprox.(θi, b; atol=ϵ)
    any(near.(θ, domain.bounds[1], ϵs)) && return true
    any(near.(θ, domain.bounds[2], ϵs)) && return true
    isnothing(domain.cons) || isapprox(domain.cons(θ), 0.; atol=cons_atol) && return true
    return false
end

function is_new_mode(θ::AbstractVector{<:Real}, keep::AbstractVector{<:AbstractVector{<:Real}}, ϵs::AbstractVector{<:Float64})
    for θ_ in keep
        any(abs.(θ .- θ_) .> ϵs) || return false
    end
    return true
end

euclidean(a, b) = sqrt(sum((b .- a) .^ 2))

function laplace_approx(logpost, μ, bounds; ϵ=0.)
    # @info "Laplace approx. for sampling: μ = $μ"

    second_derivative = ForwardDiff.jacobian(x -> ForwardDiff.gradient(logpost, x), μ)
    if any(isnan.(second_derivative)) || any(isinf.(second_derivative))
        @warn "Laplace approx. failed! NaNs in the second derivative."
        # return nothing
        @warn "Falling back on `Σ = ((ub .- lb) ./ 5) * I`."
        lb, ub = bounds
        Σ = Diagonal((ub .- lb) ./ 5)
    else
        Σ = inv((-1) * second_derivative)
    end

    # solve numerical issues
    #   - Σ should already be symmetric, but might not be due to small numerical imprecisions
    Σ = Symmetric(Σ) + (ϵ * I) # add a small diagonal matrix to ensure positive definiteness
    
    if !isposdef(Σ)
        @warn "Laplace approx. failed! Covariance matrix is not positive definite."
        # return nothing
        @warn "Falling back on `Σ = ((ub .- lb) ./ 5) * I`."
        lb, ub = bounds
        Σ = Diagonal((ub .- lb) ./ 5)
    end

    return MvNormal(μ, Σ)
end
