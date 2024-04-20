module BOLFI

export bolfi!
export BolfiProblem

export posterior_mean, posterior_variance

export BolfiAcquisition
export PostVariance

using BOSS
using Distributions

include("problem.jl")
include("posterior.jl")
include("acquisition.jl")
include("main.jl")

end
