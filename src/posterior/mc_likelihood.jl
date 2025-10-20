
function log_likelihood_mean(like::MonteCarloLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = _sample_E(y_dim(bosip), mc_samples(like))
    return _integrate_over_delta(model_post, (δ, x) -> loglike(like, δ, x), E)
end
function log_sq_likelihood_mean(like::MonteCarloLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = _sample_E(y_dim(bosip), mc_samples(like))
    return _integrate_over_delta(model_post, (δ, x) -> 2 * loglike(like, δ, x), E)
end

function log_likelihood_variance(like::MonteCarloLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    E = _sample_E(y_dim(bosip), mc_samples(like))
    
    function log_like_var(x::AbstractVector{<:Real})
        μ_δ, σ_δ = mean_and_std(model_post, x)
        
        # Monte Carlo with direct variance definition
        δ_samples = μ_δ .+ σ_δ .* E
        ll_samples = loglike.(Ref(like), eachcol(δ_samples), Ref(x))
        
        # Convert to ψ values and compute Var[ψ] = E[(ψ - E[ψ])²]
        ψ_samples = exp.(ll_samples)
        mean_ψ = mean(ψ_samples)
        var_ψ = mean((ψ_samples .- mean_ψ).^2)
        return log(var_ψ)
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end

function _sample_E(dim::Int, n::Int)
    return rand(MvNormal(zeros(dim), ones(dim)), n)
end

function _integrate_over_delta(
    model_post::ModelPosterior,
    log_func::Function,
    E::AbstractMatrix{<:Real}
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
