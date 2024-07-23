module BOLFI

export bolfi!
export BolfiProblem

export approx_posterior, posterior_mean, posterior_variance, evidence
export gp_mean, gp_bound, gp_quantile
export find_cutoff, approx_cutoff_area
export get_subset
export set_iou

export BolfiAcquisition, PostVarAcq, MWMVAcq
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback
export BolfiOptions

using BOSS
using Distributions

include("problem.jl")
include("subset_problem.jl")
include("utils.jl")
include("options.jl")
include("posterior.jl")
include("confidence_set.jl")
include("acquisition.jl")
include("term_cond.jl")
include("main.jl")

end
