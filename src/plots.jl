
"""
    PlotSettings(; kwargs...)

Aggregates all plot settings for the `plot_marginals_int` and `plot_marginals_kde` functions.

# Kwargs
- `grid_size::Int64`: The size of the grid used for plotting. 
        This hyperparameter greatly impacts the computational cost of the plotting.
- `param_labels::Union{Nothing, Vector{String}}`: Labels used for the parameters in the plots.
        Defaults to `nothing`, in which case the default labels "x1,x2,..." are used.
- `plot_data::Bool`: Controls whether to plot the datapoints queried from the simulation
        (including the initial data).
- `plot_samples::Bool`: Only relevant to `plot_marginals_kde`. Controls whether the samples
        from the posterior are also plotted.
- `full_matrix::Bool`: Set to `full_matrix=false` to only plot the marginals below the diagonal
        and skipped the redundant mirrored plots above the diagonal.
- `plot_cell_res::Int64`: The resolution (in pixels) of a single plot in the plot matrix.
"""
abstract type PlotSettings end

"""
    using CairoMakie
    plot_marginals_int(::BolfiProblem; kwargs...)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by numerically integrating the marginal integrals
over a generated latin hypercube grid of parameter samples.
The plots are normalized according to the plotting grid.

Also provides an option to plot "marginals" of different functions by using the `func` and `normalize` keywords.

# Kwargs

- `func::Function`: Defines the function which is plotted.
        The plotted function `f` is defined as `f = func(::BolfiProblem)`.
        Reasonable options for `func` include `approx_posterior`, `posterior_mean`, `posterior_variance` etc.
- `normalize::Bool`: Specifies whether the plotted marginals are normalized.
        If `normalize=false`, the plotted values are simply averages over the random LHC grid.
        If `normalize=true`, the plotted values are additionally normalized sum to 1.
        Defaults to `true`.
- `lhc_grid_size::Int`: The number of samples in the generate LHC grid.
        The higher the number, the more precise marginal plots.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.
- `matrix_ops::Bool`: Set to `false` to disable the use of matrix operations
        for plotting the marginals is they are not supported for the given `func`.
        Disabling matrix operations can significantly hinder performance.
"""
function plot_marginals_int end

"""
    using CairoMakie, Turing
    plot_marginals_kde(::BolfiProblem; kwargs...)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by kernel density estimation
over parameter samples drawn by MCMC methods from the Turing.jl package.
The plots are normalized according to the plotting grid.

One should experiment with different kernel length-scales to obtain a good approximation
of the marginals. The kernel and length-scales are provided via the `kernel` and `lengthscale`
keyword arguments.

# Kwargs

- `turing_options::TuringOptions`: Settings for the MCMC sampling.
- `kernel::Kernel`: The kernel used in the KDE.
- `lengthscale::Union{<:Real, <:AbstractVector{<:Real}}`: The lengthscale for the kernel used in the KDE.
        Either provide a single length-scale used for all parameter dimensions as a real number,
        or provide individual length-scales for each parameter dimension as a vector of real numbers.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.
"""
function plot_marginals_kde end

# The plotting is implemented in the `CairoExt` extension.
