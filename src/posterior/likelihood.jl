
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

# The default method for `Likelihood`s implementing `sq_likelihood_mean`
# instead of `likelihood_variance`.
function log_likelihood_variance(like::Likelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    log_like_mean = log_likelihood_mean(like, bolfi, model_post)
    log_sq_like_mean = log_sq_likelihood_mean(like, bolfi, model_post)

    function log_like_var(x::AbstractVector{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        return log( exp(log_sq_like_mean(x)) - exp(2 * log_like_mean(x)) )
    end
end
