const MAX_NEG_VAR = 1e-8

function log_approx_likelihood(like::Likelihood, model_post::ModelPosterior)
    function log_approx_like(x::AbstractVector{<:Real})
        μy = mean(model_post, x)
        return loglike(like, μy, x)
    end
    function log_approx_like(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)
        return loglike(like, μY, X)
    end
    return log_approx_like
end

# The default method for `Likelihood`s implementing `sq_likelihood_mean`
# instead of `likelihood_variance`.
function log_likelihood_variance(like::Likelihood, model_post::ModelPosterior)
    log_like_mean = log_likelihood_mean(like, model_post)
    log_sq_like_mean = log_sq_likelihood_mean(like, model_post)

    function log_like_var(x::AbstractArray{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        log_sqL = log_sq_like_mean(x)
        log_L = log_like_mean(x)
        like_var = @. exp(log_sqL) - exp(2 * log_L)
        like_var = _assure_nonneg.(like_var)
        return log.(like_var)
    end
end

function _assure_nonneg(x::Real)
    (x >= 0) && return x
    (x >= -MAX_NEG_VAR) && return 0.0
    throw(DomainError(x, "Expected a non-negative value, but got $x."))
end
