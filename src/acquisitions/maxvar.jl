
"""
    MaxVar()

Selects the new evaluation point by maximizing the variance of the posterior approximation.
"""
struct MaxVar <: BolfiAcquisition end

function (acq::MaxVar)(::Type{<:UniFittedParams}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return posterior_variance(bolfi; normalize=false)
end

# Use the specialized `posterior_variance` computation defined for Bayesian inference.
# This is NOT equivalent to the fallback behavior defined in
#   `(acq::BolfiAcquisition)(::Type{<:MultiFittedParams}, bolfi::BolfiProblem, options::BolfiOptions)`.
function (acq::MaxVar)(::Type{<:MultiFittedParams}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return posterior_variance(bolfi; normalize=false)
end
