module BOLFI

export bolfi!
export estimate_parameters!, maximize_acquisition, eval_objective!
export BolfiProblem
export x_dim, y_dim

export approx_posterior, posterior_mean, posterior_variance
export approx_likelihood, likelihood_mean, likelihood_variance
export log_approx_posterior, log_posterior_mean, log_posterior_variance
export log_approx_likelihood, log_likelihood_mean, log_likelihood_variance
export evidence
export loglike, like
export sample_approx_posterior, sample_expected_posterior, sample_posterior, resample
export calculate_metric
export plot_marginals_int, plot_marginals_kde, PlotSettings
export find_cutoff, approx_cutoff_area, set_iou
export get_subset
export approx_by_gauss_mix, GaussMixOptions

export Likelihood
export ExpLikelihood, SqExpLikelihood
export GutmannNormalLikelihood, GutmannGaussianLikelihood
export NormalLikelihood, GaussianLikelihood
export LogNormalLikelihood, LogGaussianLikelihood
export BinomialLikelihood

export BolfiAcquisition, PostVarAcq, MWMVAcq
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback, CombinedCallback, NoCallback
export BolfiOptions

export DistributionSampler, TuringSampler, RejectionSampler, AMISSampler
export LikelihoodMaximizer
export ProposalDistribution, NormalProposal
export DistributionFitter, AnalyticalFitter, OptimizationFitter

export DistributionMetric, SampleMetric, PDFMetric
export MMDMetric
export TVMetric

using BOSS
using Distributions
using LinearAlgebra
using Random
using KernelFunctions
using Statistics
using Optimization
using ProgressMeter
using ForwardDiff

using StatsFuns         # used in GutmannNormalLikelihood
using SpecialFunctions  # used in utils/owent.jl: erfc

import BOSS.x_dim
import BOSS.y_dim

include("include.jl")

end
