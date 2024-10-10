module BOLFI

export bolfi!
export BolfiProblem

export approx_posterior, posterior_mean, posterior_variance, evidence
export gp_mean, gp_bound, gp_quantile
export find_cutoff, approx_cutoff_area
export get_subset
export set_iou

export BolfiAcquisition, PostVarAcq, MWMVAcq, InfoGain
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback
export BolfiOptions

using BOSS
using Distributions
using LinearAlgebra
using Random

include("problem/include.jl")
include("utils.jl")
include("options.jl")
include("posterior.jl")
include("confidence_set.jl")
include("acquisition/include.jl")
include("term_cond/include.jl")
include("main.jl")

end
