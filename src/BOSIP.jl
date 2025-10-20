module BOSIP

export bosip!
export estimate_parameters!, maximize_acquisition, eval_objective!
export BosipProblem
export x_dim, y_dim

export approx_posterior, posterior_mean, posterior_variance
export approx_likelihood, likelihood_mean, likelihood_variance
export log_approx_posterior, log_posterior_mean, log_posterior_variance
export log_approx_likelihood, log_likelihood_mean, log_likelihood_variance
export evidence
export loglike, like
export construct_acquisition
export sample_approx_posterior, sample_expected_posterior
export sample_posterior, sample_posterior_pure, resample
export calculate_metric
export plot_marginals_int, plot_marginals_kde, PlotSettings
export find_cutoff, approx_cutoff_area, set_iou
export get_subset
export approx_by_gauss_mix, GaussMixOptions

export Likelihood, MonteCarloLikelihood, CustomLikelihood
export IdentityLikelihood
export ExpLikelihood, SqExpLikelihood
export GutmannNormalLikelihood, GutmannGaussianLikelihood
export NormalLikelihood, GaussianLikelihood
export LogNormalLikelihood
export BinomialLikelihood
export MvNormalLikelihood
export NormalSumLikelihood
export LogNormalSumLikelihood
export NormalDiffLikelihood

export BosipAcquisition, MaxVar, LogMaxVar, MWMV, IMMD, EIV, IMIQR
export BosipTermCond, AEConfidence, UBLBConfidence
export BosipCallback, CombinedCallback, NoCallback, MetricCallback
export BosipOptions

export DistributionSampler, PureSampler, WeightedSampler
export RejectionSampler, TuringSampler, AMISSampler
export LogpdfMaximizer
export ProposalDistribution, NormalProposal
export DistributionFitter, AnalyticalFitter, OptimizationFitter

export DistributionMetric, SampleMetric, PDFMetric
export MMDMetric, OptMMDMetric
export TVMetric

using BOSS
using Distributions
using LinearAlgebra
using Random
using KernelFunctions
using Statistics
using Optimization
using ProgressMeter
using DifferentiationInterface
using ForwardDiff
using LazyArrays

# used in utils/owent.jl (erfc)
using SpecialFunctions

import BOSS.x_dim, BOSS.y_dim
import BOSS.estimate_parameters!, BOSS.maximize_acquisition, BOSS.eval_objective!
import BOSS.construct_acquisition

include("include.jl")

end
