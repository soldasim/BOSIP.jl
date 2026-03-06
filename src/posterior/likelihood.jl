# Default implementations for the `Likelihood` API.

const MAX_NEG_VAR = 1e-8

# Defined by `loglike_marginal`
function log_approx_marginal_likelihood(like::Likelihood, model_post::ModelPosterior)
    function log_approx_ml(x::AbstractVector{<:Real})
        μy = mean(model_post, x)
        return loglike_marginal(like, μy, x)
    end
    function log_approx_ml(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)
        return loglike_marginal(like, μY, X)
    end
    return log_approx_ml
end

# Defined by `loglike`
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

# Defined by `log_marginal_likelihood_mean`
# Expectation of a product is equivalent to the product of expectation as long as the individual quantities are independent.
# Our individual proxy dimensions are modeled independently, so this holds.
function log_likelihood_mean(like::Likelihood, model_post::ModelPosterior)
    log_ml_mean = log_marginal_likelihood_mean(like, model_post)
    function log_like_mean(x::AbstractVector{<:Real})
        return sum(log_ml_mean(x))
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return vec(sum(log_ml_mean(X), dims=1))
    end
    return log_like_mean
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
