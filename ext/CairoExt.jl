module CairoExt

using BOSIP, BOSS
using CairoMakie
using KernelFunctions
using Statistics
using LinearAlgebra
using Distributions
using OptimizationPRIMA
using ProgressMeter

@kwdef struct PlotSettings <: BOSIP.PlotSettings
    grid_size::Int = 200
    resolution::Int = 600
    param_labels::Union{Nothing, Vector{String}} = nothing
    plot_data::Bool = true
    plot_samples::Bool = true
    plot_pair_marginals::Bool = true
    plot_diagonal::Bool = true
    full_matrix::Bool = true
    upper_triangle::Bool = true
    x_true::Union{Nothing, AbstractVector{<:Real}} = nothing
    title::Union{Nothing, String} = nothing
    plot_style::Symbol = :heatmap
    colormap::Symbol = :matter
end

BOSIP.PlotSettings(args...; kwargs...) = PlotSettings(args...; kwargs...)

@kwdef struct MarginalData{N}
    xs::NTuple{N, Vector{Float64}}
    ys::AbstractArray{Float64, N}
end

@kwdef struct PlotData <: BOSIP.PlotData
    marginals::Matrix{<:Union{MarginalData, Missing}}
    bounds::AbstractBounds
    X::Union{AbstractMatrix{<:Real}, Missing}
end

BOSIP.PlotData(args...; kwargs...) = PlotData(args...; kwargs...)

const SAMPLES_SCATTER_KWARGS = (
    color = :green,
    marker = :xcross,
    markersize = 3,
)
const DATA_SCATTER_KWARGS = (
    color = :grey,
    marker = :circle,
    markersize = 5,
)
const X_TRUE_VLINE_KWARGS = (
    color = :red,
    linestyle = :dash,
    linewidth = 2,
)
const X_TRUE_SCATTER_KWARGS = (
    color = :red,
    marker = :star5,
    markersize = 12,
)
const TITLE_KWARGS = (
    font = :bold,
)

_default_kde_sampler() = AMISSampler(;
    iters = 10,
    proposal_fitter = BOSIP.AnalyticalFitter(), # re-fit the proposal analytically
    gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
        algorithm = BOBYQA(),
        multistart = 24,
        parallel = true,
        cluster_ϵs = nothing,
        rel_min_weight = 1e-8,
        rhoend = 1e-4,
    ),
)

const KDE_SUPERSAMPLE = 20

function BOSIP.plot_marginals_int(bosip::BosipProblem;
    func = BOSIP.posterior_mean,
    plot_settings = PlotSettings(),
    lhc_grid_size = 200,
    matrix_ops = true,
    info = true,
    display = false,
    create_plot = true, # not intended for users
)
    info && @info "Generating a latin hypercube of parameter samples ..."
    xs = BOSS.generate_LHC(bosip.problem.domain.bounds, lhc_grid_size)
    keep = BOSS.in_domain.(eachcol(xs), Ref(bosip.problem.domain))
    info && (sum(keep) < lhc_grid_size) && @warn "Discarding some LHC points outside of the parameter domain."
    xs = xs[:, keep]

    info && @info "Computing marginals ..."
    data = _compute_marginals_int(bosip, xs; func, plot_settings, matrix_ops, info)

    if create_plot
        info && @info "Plotting ..."
        return plot_marginals(data; plot_settings, display)
    else
        return data
    end
end

function BOSIP.plot_marginals_kde(bosip::BosipProblem;
    logpost_func = BOSIP.log_posterior_mean,
    sampler = _default_kde_sampler(),
    sample_count::Int = 1000,
    kernel = GaussianKernel(),
    lengthscale = 0.2,
    plot_settings = PlotSettings(),
    info = true,
    display = false,
    create_plot = true, # not intended for users
)
    info && @info "Sampling parameter samples from the posterior ..."
    logpost = logpost_func(bosip)
    xs = sample_posterior_pure(sampler, logpost, bosip.problem.domain, sample_count;
        supersample_ratio = KDE_SUPERSAMPLE,
    )
    
    info && @info "Computing marginals ..."
    (lengthscale isa Real) && (lengthscale = fill(lengthscale, BOSIP.x_dim(bosip)))
    data = _compute_marginals_kde(bosip, xs, kernel, lengthscale; plot_settings, info)

    if create_plot
        info && @info "Plotting ..."
        return plot_marginals(data; plot_settings, display)
    else
        return data
    end
end

function BOSIP.compute_marginals_int(bosip::BosipProblem; kwargs...)
    data = BOSIP.plot_marginals_int(bosip; create_plot=false, kwargs...)
    return data
end
function BOSIP.compute_marginals_kde(bosip::BosipProblem; kwargs...)
    data = BOSIP.plot_marginals_kde(bosip; create_plot=false, kwargs...)
    return data
end

function _compute_marginals_int(bosip::BosipProblem, grid::AbstractMatrix{<:Real};
    func,
    plot_settings,
    matrix_ops,
    info = true,
)
    x_dim = BOSIP.x_dim(bosip)
    count = size(grid)[2]
    bounds = bosip.problem.domain.bounds
    f = func(bosip)
    X = bosip.problem.data.X
    grid_size = plot_settings.grid_size
    steps = plot_steps(grid_size, bounds)

    # init data
    data = PlotData(;
        marginals = Matrix{Union{MarginalData, Missing}}(missing, x_dim, x_dim),
        bounds = bounds,
        X = X,
    )

    tmp_row_a = zeros(count)
    tmp_row_b = zeros(count)

    # single parameter marginals (diagonal)
    if plot_settings.plot_diagonal
        dims = 1:x_dim

        info && (prog = Progress(length(dims) * grid_size; desc="Diagonal marginals: "))
        for dim in dims
            tmp_row_a .= grid[dim,:]

            xs = range(bounds[1][dim], bounds[2][dim]; length=grid_size)
            ys = zeros(length(xs))
            
            for (i, x_) in enumerate(xs)
                grid[dim,:] .= x_

                if matrix_ops
                    ys[i] = mean(f(grid))
                else
                    ys[i] = mean(f.(eachcol(grid)))
                end
                info && next!(prog)
            end
            normalize_prob_vals!(ys, steps[dim])

            # store data
            data.marginals[dim, dim] = MarginalData(;
                xs = (collect(xs),),
                ys = ys,
            )

            grid[dim,:] .= tmp_row_a
        end
    end

    # pair marginals
    if plot_settings.plot_pair_marginals
        pairs = [(dim_a, dim_b) for dim_a in 1:x_dim for dim_b in dim_a+1:x_dim]

        info && (prog = Progress(length(pairs) * grid_size^2; desc="Pairwise marginals: "))
        for (dim_a, dim_b) in pairs
            tmp_row_a .= grid[dim_a,:]
            tmp_row_b .= grid[dim_b,:]

            xs_a = range(bounds[1][dim_a], bounds[2][dim_a]; length=grid_size)
            xs_b = range(bounds[1][dim_b], bounds[2][dim_b]; length=grid_size)
            xs = Iterators.product(xs_a, xs_b)
            ys = zeros(size(xs))
            
            for (i, x_) in enumerate(xs)
                grid[dim_a,:] .= x_[1]
                grid[dim_b,:] .= x_[2]

                if matrix_ops
                    ys[i] = mean(f(grid))
                else
                    ys[i] = mean(f.(eachcol(grid)))
                end
                info && next!(prog)
            end
            normalize_prob_vals!(ys, steps[dim_a], steps[dim_b])

            # store data
            data.marginals[dim_a, dim_b] = MarginalData(;
                xs = (collect(xs_a), collect(xs_b)),
                ys = ys,
            )
            data.marginals[dim_b, dim_a] = MarginalData(;
                xs = (collect(xs_b), collect(xs_a)),
                ys = ys',
            )

            grid[dim_a,:] .= tmp_row_a
            grid[dim_b,:] .= tmp_row_b
        end
    end

    return data
end

function _compute_marginals_kde(bosip::BosipProblem, samples::AbstractMatrix{<:Real}, kernel::Kernel, lengthscale::AbstractVector{<:Real};
    plot_settings,
    info = true,
)
    x_dim = BOSIP.x_dim(bosip)
    bounds = bosip.problem.domain.bounds
    X = bosip.problem.data.X
    grid_size = plot_settings.grid_size
    steps = plot_steps(grid_size, bounds)

    # init data
    data = PlotData(;
        marginals = Matrix{Union{MarginalData, Missing}}(missing, x_dim, x_dim),
        bounds = bounds,
        X = X,
    )

    # single parameter marginals (diagonal)
    if plot_settings.plot_diagonal
        dims = 1:x_dim

        info && (prog = Progress(length(dims) * grid_size; desc="Diagonal marginals: "))
        for dim in dims
            xs = range(bounds[1][dim], bounds[2][dim]; length=grid_size) |> collect
            ys = zeros(length(xs))
            
            k = with_lengthscale(kernel, lengthscale[dim])
            for i in eachindex(xs)
                ys[i] = mean(k.(Ref(xs[i]), samples[dim,:]))
                info && next!(prog)
            end
            normalize_prob_vals!(ys, steps[dim])

            # store data
            data.marginals[dim, dim] = MarginalData(;
                xs = (xs,),
                ys = ys,
            )
        end
    end

    # pair marginals
    if plot_settings.plot_pair_marginals
        pairs = [(dim_a, dim_b) for dim_a in 1:x_dim for dim_b in dim_a+1:x_dim]

        info && (prog = Progress(length(pairs) * grid_size^2; desc="Pairwise marginals: "))
        for (dim_a, dim_b) in pairs
            xs_a = range(bounds[1][dim_a], bounds[2][dim_a]; length=grid_size)
            xs_b = range(bounds[1][dim_b], bounds[2][dim_b]; length=grid_size)
            xs = Iterators.product(xs_a, xs_b) |> collect .|> (t -> [t...])
            ys = zeros(size(xs))
            
            k = with_lengthscale(kernel, lengthscale[[dim_a,dim_b]])
            for i in eachindex(xs)
                ys[i] = mean(k.(Ref(xs[i]), eachcol(samples[[dim_a,dim_b],:])))
                info && next!(prog)
            end
            normalize_prob_vals!(ys, steps[dim_a], steps[dim_b])

            # store data
            data.marginals[dim_a, dim_b] = MarginalData(;
                xs = (collect(xs_a), collect(xs_b)),
                ys = ys,
            )
            data.marginals[dim_b, dim_a] = MarginalData(;
                xs = (collect(xs_b), collect(xs_a)),
                ys = ys',
            )
        end
    end

    return data
end

function BOSIP.plot_marginals(data::PlotData;
    plot_settings = PlotSettings(),
    display = false,
)
    x_dim = size(data.X, 1)
    X = data.X

    fig = Figure(;
        size = (plot_settings.resolution, plot_settings.resolution),
    )
    labels = isnothing(plot_settings.param_labels) ? ["x$i" for i in 1:x_dim] : plot_settings.param_labels
    @assert length(labels) == x_dim

    # single parameter marginals (diagonal)
    if plot_settings.plot_diagonal
        dims = 1:x_dim

        for dim in dims
            xs = data.marginals[dim, dim].xs |> first
            ys = data.marginals[dim, dim].ys
            
            ax = Axis(fig[dim,dim]; xlabel=labels[dim])
            lines!(ax, xs, ys)
            if plot_settings.plot_data
                @assert !ismissing(X)
                scatter!(ax, X[dim,:], zeros(size(X)[2]);
                    DATA_SCATTER_KWARGS...
                )
            end
            isnothing(plot_settings.x_true) || vlines!(ax, [plot_settings.x_true[dim]];
                X_TRUE_VLINE_KWARGS...
            )
        end
    end

    # pair marginals
    if plot_settings.plot_pair_marginals
        plot_func = getproperty(CairoMakie, Symbol(string(plot_settings.plot_style) * '!'))
        pairs = [(dim_a, dim_b) for dim_a in 1:x_dim for dim_b in dim_a+1:x_dim]

        for (dim_a, dim_b) in pairs
            @assert !ismissing(data.marginals[dim_a, dim_b])
            xs_a, xs_b = data.marginals[dim_a, dim_b].xs
            ys = data.marginals[dim_a, dim_b].ys

            if plot_settings.full_matrix || !plot_settings.upper_triangle
                ax = Axis(fig[dim_b, dim_a]; xlabel=labels[dim_a], ylabel=labels[dim_b])
                plot_func(ax, xs_a, xs_b, ys; plot_settings.colormap)
                if plot_settings.plot_data
                    @assert !ismissing(X)
                    scatter!(ax, X[dim_a,:], X[dim_b,:];
                        DATA_SCATTER_KWARGS...
                    )
                end
                isnothing(plot_settings.x_true) || scatter!(ax, [plot_settings.x_true[dim_a]], [plot_settings.x_true[dim_b]];
                    X_TRUE_SCATTER_KWARGS...
                )
            end

            if plot_settings.full_matrix || plot_settings.upper_triangle
                ax_t = Axis(fig[dim_a, dim_b]; xlabel=labels[dim_b], ylabel=labels[dim_a])
                plot_func(ax_t, xs_b, xs_a, ys'; plot_settings.colormap)
                if plot_settings.plot_data
                    @assert !ismissing(X)
                    scatter!(ax_t, X[dim_b,:], X[dim_a,:];
                        DATA_SCATTER_KWARGS...
                    )
                end
                isnothing(plot_settings.x_true) || scatter!(ax_t, [plot_settings.x_true[dim_b]], [plot_settings.x_true[dim_a]];
                    X_TRUE_SCATTER_KWARGS...
                )
            end
        end
    end

    # eliminate empty cols/rows
    trim!(fig.layout)
    # fix for single col figures properly scaling to full width
    (length(fig.content) == 1) && colsize!(fig.layout, 1, Relative(1))

    if !isnothing(plot_settings.title)
        fig[0,:] = Label(fig, plot_settings.title; TITLE_KWARGS...)
    end

    display && CairoMakie.display(fig)
    return fig
end

function normalize_prob_vals!(ys::AbstractVector{<:Real}, step::Real)
    total = sum(ys)
    total -= 0.5 * (ys[begin] + ys[end])
    
    if total == 0.
        ys .= 1. / (step * (length(ys) - 1))
    else
        ys ./= step * total
    end
end
function normalize_prob_vals!(ys::AbstractMatrix{<:Real}, step_a::Real, step_b::Real)
    total = sum(ys)
    total -= 0.5 * (sum(ys[begin,:]) + sum(ys[end,:]) + sum(ys[:,begin]) + sum(ys[:,end]))
    total += 0.25 * (ys[begin,begin] + ys[begin,end] + ys[end,begin] + ys[end,end])
    
    if total == 0.
        ys .= 1. / (step_a * step_b * (size(ys,1) - 1) * (size(ys,2) - 1))
    else
        ys ./= step_a * step_b * total
    end
end

function plot_steps(grid::Int, bounds::AbstractBounds)
    lb, ub = bounds
    return (ub .- lb) ./ (grid - 1)
end

end # module CairoExt
