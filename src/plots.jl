
"""
    PlotSettings(; kwargs...)

Aggregates all plot settings for the `plot_marginals_int` and `plot_marginals_kde` functions.

## Kwargs
- `grid_size::Int64`: The size of the grid used for plotting. 
        This hyperparameter greatly impacts the computational cost of the plotting.
- `resolution::Int`: The resolution of the individual plots in the marginal matrix.
- `param_labels::Union{Nothing, Vector{String}}`: Labels used for the parameters in the plots.
        Defaults to `nothing`, in which case the default labels "x1,x2,..." are used.
- `plot_data::Bool`: Controls whether to plot the datapoints queried from the simulation
        (including the initial data).
- `plot_samples::Bool`: Only relevant to `plot_marginals_kde`. Controls whether the samples
        from the posterior are also plotted.
- `plot_pair_marginals::Bool`: Set to `false` to skip plotting the pairwise parameter marginals
        in the off-diagonal of the marginal matrix.
- `plot_diagonal::Bool`: Set to `diagonal=false` to skip plotting the univariate parameter marginals
        on the diagonal of the marginal matrix.
- `full_matrix::Bool`: Set to `full_matrix=false` to only plot the marginals below the diagonal
        and skipped the redundant mirrored plots above the diagonal.
- `upper_triangle::Bool` Switch between plotting the lower triangle (`upper_triangle=false`)
        or the upper triangle (`upper_triangle=true`) of the marginal matrix. Only has
        an effect if `full_matrix=false`.
- `x_true::Union{Nothing, AbstractVector{<:Real}}`: The true parameter values
        to be indicated in the plots.
- `title::Union{Nothing, String}`: A plot title.
- `plot_style::Symbol`: The style of the 2d plots. Can be `:heatmap`, :contour, or `:contourf`.
- `colormap::Symbol`: The colormap used for the 2d plots. We recommend trying `:matter`, `:viridis`, or `:plasma`.
"""
abstract type PlotSettings end

"""
    PlotData

Stores all precomputed data required for plotting the marginals.

Instances of `PlotData` can be obtained via the `compute_marginals_int` and `compute_marginals_kde` functions
and the plots can be created later via the `plot_marginals` function.

Direct instantiation of `PlotData` by the user is not intended.
"""
abstract type PlotData end

"""
    using CairoMakie
    fig = plot_marginals_int(::BosipProblem; kwargs...)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by numerically integrating the marginal integrals
over a generated latin hypercube grid of parameter samples.
The plots are normalized according to the plotting grid.

It is necessary to adapt the number of samples (defined through the `lhc_grid_size` keyword)
according to the dimensionality of the parameter space.

Also provides an option to plot "marginals" of different functions by using the `func` and `normalize` keywords.

## Kwargs

- `func::Function`: Defines the function which is plotted.
        The plotted function `f` is defined as `f = func(::BosipProblem)`.
        Reasonable options for `func` include `approx_posterior`, `posterior_mean`, `posterior_variance` etc.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `lhc_grid_size::Int`: The number of samples in the generate LHC grid.
        The higher the number, the more precise marginal plots.
        This settings affects the computational cost of the plotting significantly.
- `matrix_ops::Bool`: Set to `false` to disable the use of matrix operations
        for plotting the marginals is they are not supported for the given `func`.
        Disabling matrix operations can significantly hinder performance.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.

## See Also
[`plot_marginals_kde`](@ref), [`compute_marginals_int`](@ref), [`PlotSettings`](@ref)
"""
function plot_marginals_int end

"""
    using CairoMakie
    fig = plot_marginals_kde(::BosipProblem; kwargs...)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by kernel density estimation
over parameter samples drawn by MCMC methods from the Turing.jl package.
The plots are normalized according to the plotting grid.

It is necessary to adapt the number of samples (defined through the `turing_options` keyword)
according to the dimensionality of the parameter space.

One should experiment with different `kernel`s, `lengthscale`s, and `sample_count`s
to obtain a good approximation of the marginals. See keyword arguments below.

## Kwargs
- `logpost_func::Function`: A function, which takes the `BosipProblem` and returns
    a function `x -> log p(x|z_obs)` computing the log-posterior density.
    Reasonable options include `log_posterior_mean` and `log_approx_posterior`.
- `sampler::DistributionSampler`: The sampler used to draw samples from the posterior.
        Defaults to the `AMISSampler`.
- `sample_count::Int`: The number of samples to draw from the posterior for the KDE.
        The higher the number, the more precise marginal plots.
        This settings affects the computational cost of the plotting significantly.
- `kernel::Kernel`: The kernel used in the KDE.
- `lengthscale::Union{<:Real, <:AbstractVector{<:Real}}`: The lengthscale for the kernel used in the KDE.
        Either provide a single length-scale used for all parameter dimensions as a real number,
        or provide individual length-scales for each parameter dimension as a vector of real numbers.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.

## See Also
[`plot_marginals_int`](@ref), [`compute_marginals_kde`](@ref), [`PlotSettings`](@ref)
"""
function plot_marginals_kde end

"""
    data = compute_marginals_int(::BosipProblem; kwargs...)

Same as [`plot_marginals_int`](@ref), but only computes and returns the `PlotData`
used for later plotting via [`plot_marginals`](@ref) instead of creating the plots immediately.

Intended use is to precompute the `PlotData`
and serialize them (e.g. via JLD2.jl) for later plotting.

See the documentation of [`plot_marginals_int`](@ref) for the keyword arguments.

## See Also
[`plot_marginals_int`](@ref), [`plot_marginals`](@ref)], [`PlotData`](@ref), [`PlotSettings`](@ref)
"""
function compute_marginals_int end

"""
    data = compute_marginals_kde(::BosipProblem; kwargs...)

Same as [`plot_marginals_kde`](@ref), but only computes and returns the `PlotData`
used for later plotting via [`plot_marginals`](@ref) instead of creating the plots immediately.

Intended use is to precompute the `PlotData`
and serialize them (e.g. via JLD2.jl) for later plotting.

See the documentation of [`plot_marginals_kde`](@ref) for the keyword arguments.

## See Also
[`plot_marginals_kde`](@ref), [`plot_marginals`](@ref)], [`PlotData`](@ref), [`PlotSettings`](@ref)
"""
function compute_marginals_kde end

"""
    fig = plot_marginals(data::PlotData; kwargs...)

Create the marginal plots from previously computed `PlotData`
via [`compute_marginals_int`](@ref) or [`compute_marginals_kde`](@ref).

## Kwargs
- `plot_settings::PlotSettings`: Settings for the plotting.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.

## See Also
[`compute_marginals_int`](@ref), [`compute_marginals_kde`](@ref), [`PlotData`](@ref), [`PlotSettings`](@ref)
"""
function plot_marginals end

# The plotting is implemented in the `CairoExt` extension.
