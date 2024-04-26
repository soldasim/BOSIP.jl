module BOLFI

export bolfi!
export BolfiProblem

export get_subset
export posterior_mean, posterior_variance, evidence
export find_cutoff, find_cutoff_confint, gp_quantile

export BolfiAcquisition
export PostVariance, SetsPostVariance

export BolfiTermCond
export ConfidenceTermCond

using BOSS
using Distributions

include("problem.jl")
include("subset_problem.jl")
include("posterior.jl")
include("confidence_set.jl")
include("acquisition.jl")
include("term_cond.jl")
include("main.jl")

end
