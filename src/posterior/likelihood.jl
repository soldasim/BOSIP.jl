
function log_approx_likelihood(like::Likelihood, bosip::BosipProblem, model_post::ModelPosterior)
    function log_approx_like(x::AbstractVector{<:Real})
        μy = mean(model_post, x)
        return loglike(like, μy)
    end
    function log_approx_like(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)
        return loglike(like, μY')
    end
    return log_approx_like
end

# The default method for `Likelihood`s implementing `sq_likelihood_mean`
# instead of `likelihood_variance`.
function log_likelihood_variance(like::Likelihood, bosip::BosipProblem, model_post::ModelPosterior)
    log_like_mean = log_likelihood_mean(like, bosip, model_post)
    log_sq_like_mean = log_sq_likelihood_mean(like, bosip, model_post)

    function log_like_var(x::AbstractVector{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        return log( exp(log_sq_like_mean(x)) - exp(2 * log_like_mean(x)) )
    end
end
