
function log_approx_likelihood(like::Likelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    function log_approx_like(x::AbstractVector{<:Real})
        μ_y = mean(model_post, x)
        return loglike(like, μ_y)
    end
    function log_approx_like(X::AbstractMatrix{<:Real})
        μs_y = mean(model_post, X)
        return loglike.(Ref(like), eachrow(μs_y))
    end
    return log_approx_like
end
