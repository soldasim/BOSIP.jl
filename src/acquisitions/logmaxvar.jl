
"""
    LogMaxVar()

Selects the new evaluation point by maximizing the log variance of the posterior approximation.

The `LogMaxVar` acquisition is functionally equivalent to `MaxVar`.
Using `MaxVar` or `LogMaxVar` can be more/less suitable in different scenarios.
Switching between the two can help with numerical stability.
"""
struct LogMaxVar <: BolfiAcquisition end

function (acq::LogMaxVar)(::Type{<:UniFittedParams}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return log_posterior_variance(bolfi)
end
