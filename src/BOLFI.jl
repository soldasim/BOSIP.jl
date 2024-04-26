module BOLFI

export bolfi!
export BolfiProblem

export posterior_mean, posterior_variance
export get_subset

export BolfiAcquisition
export PostVariance, SetsPostVariance

using BOSS
using Distributions

include("problem.jl")
include("posterior.jl")
include("acquisition.jl")
include("main.jl")
include("user_utils.jl")

end
