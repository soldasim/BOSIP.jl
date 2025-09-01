
"""
    MaxVar()

Selects the new evaluation point by maximizing the variance of the posterior approximation.
"""
struct MaxVar <: BosipAcquisition end

function (acq::MaxVar)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, options::BosipOptions)
    return posterior_variance(bosip; normalize=false)
end

# Use the specialized `posterior_variance` computation defined for Bayesian inference.
# This is NOT equivalent to the fallback behavior defined in
#   `(acq::BosipAcquisition)(::Type{<:MultiFittedParams}, bosip::BosipProblem, options::BosipOptions)`.
function (acq::MaxVar)(::Type{<:MultiFittedParams}, bosip::BosipProblem{Nothing}, options::BosipOptions)
    return posterior_variance(bosip; normalize=false)
end
