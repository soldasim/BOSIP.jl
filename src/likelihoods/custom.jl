
"""
    CustomLikelihood(; log_ψ::Function)

## Keywords
- `log_ψ::Function`: A function `log(ℓ) = log_ψ(δ, x)` computing the log-likelihood
        for a given model output `δ` and input parameters `x`.
        Here, `δ` is the proxy variable modeled by the surrogate model
        and `x` are the input parameters (which will usually not be used for the calculation).
- `mc_samples::Int = 1000`: Number of Monte Carlo samples to use when computing the expected log-likelihood
        and its variance.
"""
@kwdef struct CustomLikelihood <: Likelihood
    log_ψ::Function
    mc_samples::Int
end

function loglike(like::CustomLikelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return like.log_ψ(δ, x)
end

function log_likelihood_mean(like::CustomLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = sample_E(y_dim(bosip), like.mc_samples)
    return integrate_over_delta(like, bosip, model_post, like.log_ψ, E)
end
function log_sq_likelihood_mean(like::CustomLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = sample_E(y_dim(bosip), like.mc_samples)
    return integrate_over_delta(like, bosip, model_post, (δ, x) -> 2 * like.log_ψ(δ, x), E)
end

function log_likelihood_variance(like::CustomLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = sample_E(y_dim(bosip), like.mc_samples)
    
    function log_like_var(x::AbstractVector{<:Real})
        μ_δ, σ_δ = mean_and_std(model_post, x)
        
        # Monte Carlo with direct variance definition
        δ_samples = μ_δ .+ σ_δ .* E
        log_ψ_samples = like.log_ψ.(eachcol(δ_samples), Ref(x))
        
        # Convert to ψ values and compute Var[ψ] = E[(ψ - E[ψ])²]
        ψ_samples = exp.(log_ψ_samples)
        mean_ψ = mean(ψ_samples)
        var_ψ = mean((ψ_samples .- mean_ψ).^2)
        return log(var_ψ)
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end

function sample_E(dim::Int, n::Int)
    return rand(MvNormal(zeros(dim), ones(dim)), n)
end

function integrate_over_delta(
    like::CustomLikelihood,
    bosip::BosipProblem,
    model_post::ModelPosterior,
    log_func::Function,
    E::Union{AbstractMatrix{<:Real}, Nothing} = nothing
)
    function log_int(x::AbstractVector{<:Real})
        μ_δ, σ_δ = mean_and_std(model_post, x)
        
        # Monte Carlo integration
        δ_samples = μ_δ .+ σ_δ .* E
        log_samples = log_func.(eachcol(δ_samples), Ref(x))
        return log(mean(exp.(log_samples)))
    end
    function log_int(X::AbstractMatrix{<:Real})
        return log_int.(eachcol(X))
    end
    return log_int
end
