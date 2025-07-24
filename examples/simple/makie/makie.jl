module MakiePlots

using CairoMakie
using Distributions
using BOSS, BOLFI
using OptimizationPRIMA

import ..ToyProblem
import ..log_posterior_estimate


# --- CALLBACK ---

mutable struct PlotCallback <: BolfiCallback
    title::String
    iters::Int
    plot_each::Int
    display::Bool
    save_plots::Bool
    plot_dir::String
    plot_name::String
    step::Float64
    parallel::Bool
    acq_grid::Union{Nothing, Vector{Float64}}
    sampler::DistributionSampler
end
PlotCallback(;
    title = "",
    plot_each::Int = 1,
    display::Bool = true,
    save_plots::Bool = false,
    plot_dir::String = active_dir(),
    plot_name::String = "p",
    step,
    parallel = true,
    acq_grid = nothing,
    sampler = _default_sampler(),
) = PlotCallback(title, 0, plot_each, display, save_plots, plot_dir, plot_name, step, parallel, acq_grid, sampler)

function _default_sampler()
    return AMISSampler(;
        iters = 10,
        gauss_mix_options = GaussMixOptions(;
            algorithm = BOBYQA(),
        ),
    )
end

"""
Plots the state in the current iteration.
"""
function (plt::PlotCallback)(bolfi::BolfiProblem; options, first, kwargs...)
    if first
        options.info && @info "Plotting ..."
        plot_state(bolfi, nothing; plt, iter=plt.iters, kwargs...)
        return
    end
    
    plt.iters += 1
    if plt.iters % plt.plot_each == 0
        options.info && @info "Plotting ..."
        plot_state(bolfi, nothing; plt, iter=plt.iters, kwargs...)
    end
end


# --- INIT ---

init_plotting(plt::PlotCallback) =
    init_plotting(; save_plots=plt.save_plots, plot_dir=plt.plot_dir)
init_plotting(::Any) = nothing

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            rm(plot_dir, recursive=true)
        end
        mkpath(plot_dir)
    end
end


# --- PLOTS ---

function plot_state(prev_state::BolfiProblem, new_datum::Union{Nothing, <:AbstractVector{<:Real}}; plt::PlotCallback, iter::Int, kwargs...)
    @assert ToyProblem.x_dim() == 2
    f = plot_state_2d(prev_state, new_datum; plt, kwargs...)

    plt.display && display(f)
    plt.save_plots && save(plt.plot_dir * '/' * plt.plot_name * "_$iter" * ".png", f)
    return f
end

include("utils.jl")
include("2d.jl")

end # MakiePlots
