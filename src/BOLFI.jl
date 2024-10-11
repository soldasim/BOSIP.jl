module BOLFI

export bolfi!
export BolfiProblem

export approx_posterior, posterior_mean, posterior_variance, evidence
export gp_mean, gp_bound, gp_quantile
export find_cutoff, approx_cutoff_area, set_iou
export get_subset

export BolfiAcquisition, PostVarAcq, MWMVAcq, InfoGain
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback
export BolfiOptions

using BOSS
using Distributions
using LinearAlgebra
using Random

include("include.jl")

end
