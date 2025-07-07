module CairoExt

using BOLFI, BOSS
using CairoMakie
using BOLFI.KernelFunctions
using BOLFI.Statistics
using BOLFI.LinearAlgebra
using BOLFI.Distributions

@kwdef struct PlotSettings <: BOLFI.PlotSettings
    grid_size::Int = 200
    param_labels::Union{Nothing, Vector{String}} = nothing
    plot_data::Bool = true
    plot_samples::Bool = true
    full_matrix::Bool = true
    plot_cell_res::Int = 400
end

BOLFI.PlotSettings(args...; kwargs...) =
    PlotSettings(args...; kwargs...)

const SAMPLES_SCATTER_KWARGS = (
    color = :green,
    marker = :xcross,
    markersize = 3,
)
const DATA_SCATTER_KWARGS = (
    color = :red,
    marker = :circle,
    markersize = 5,
)

function BOLFI.plot_marginals_int(bolfi::BolfiProblem;
    func = approx_posterior,
    normalize = true,
    lhc_grid_size = 1000,
    plot_settings = PlotSettings(),
    info = true,
    display = true,
    matrix_ops = true,
)
    info && @info "Generating a latin hypercube of parameter samples ..."
    xs = BOSS.generate_LHC(bolfi.problem.domain.bounds, lhc_grid_size)
    keep = BOSS.in_domain.(eachcol(xs), Ref(bolfi.problem.domain))
    info && (sum(keep) < lhc_grid_size) && @warn "Discarding some LHC points outside of the parameter domain."
    xs = xs[:, keep]

    info && @info "Plotting the marginals ..."
    return plot_marginals_int(bolfi, xs; func, normalize, plot_settings, display, matrix_ops)
end

function BOLFI.plot_marginals_kde(bolfi::BolfiProblem;
    turing_options = TuringOptions(),
    kernel = GaussianKernel(),
    lengthscale = 1.,
    plot_settings = PlotSettings(),
    info = true,
    display = false,
)
    info && @info "Sampling parameter samples from the posterior ..."
    xs = sample_posterior(bolfi, turing_options)

    info && @info "Plotting the marginals ..."
    (lengthscale isa Real) && (lengthscale = fill(lengthscale, BOLFI.x_dim(bolfi)))
    return plot_marginals_kde(bolfi, xs, kernel, lengthscale; plot_settings, display)
end

function plot_marginals_int(bolfi::BolfiProblem, grid::AbstractMatrix{<:Real};
    func,
    normalize,
    plot_settings,
    display,
    matrix_ops,
    kwargs...
)
    x_dim = BOLFI.x_dim(bolfi)
    count = size(grid)[2]
    bounds = bolfi.problem.domain.bounds
    f = func(bolfi)
    X = bolfi.problem.data.X

    grid_size = plot_settings.grid_size
    steps = plot_steps(grid_size, bounds)

    res = x_dim * plot_settings.plot_cell_res
    fig = Figure(;
        size = (res, res),
    )
    labels = isnothing(plot_settings.param_labels) ? ["x$i" for i in 1:x_dim] : plot_settings.param_labels
    @assert length(labels) == x_dim

    tmp_row_a = zeros(count)
    tmp_row_b = zeros(count)

    # single parameter marginals (diagonal)
    dims = 1:x_dim

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
        end
        normalize && normalize_prob_vals!(ys, steps[dim])

        ax = Axis(fig[dim,dim]; xlabel=labels[dim])
        lines!(ax, xs, ys)
        plot_settings.plot_data && scatter!(ax, X[dim,:], zeros(size(X)[2]);
            DATA_SCATTER_KWARGS...
        )

        grid[dim,:] .= tmp_row_a
    end

    # pair marginals
    pairs = [(dim_a, dim_b) for dim_a in 1:x_dim for dim_b in dim_a+1:x_dim]

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
        end
        normalize && normalize_prob_vals!(ys, steps[dim_a], steps[dim_b])

        ax = Axis(fig[dim_b, dim_a]; xlabel=labels[dim_a], ylabel=labels[dim_b])
        contourf!(ax, xs_a, xs_b, ys)
        plot_settings.plot_data && scatter!(ax, X[dim_a,:], X[dim_b,:];
            DATA_SCATTER_KWARGS...
        )
        if plot_settings.full_matrix
            ax_t = Axis(fig[dim_a, dim_b]; xlabel=labels[dim_b], ylabel=labels[dim_a])
            contourf!(ax_t, xs_b, xs_a, ys')
            plot_settings.plot_data && scatter!(ax_t, X[dim_b,:], X[dim_a,:];
                DATA_SCATTER_KWARGS...
            )
        end

        grid[dim_a,:] .= tmp_row_a
        grid[dim_b,:] .= tmp_row_b
    end

    display && CairoMakie.display(fig)
    return fig
end

function plot_marginals_kde(bolfi::BolfiProblem, samples::AbstractMatrix{<:Real}, kernel::Kernel, lengthscale::AbstractVector{<:Real};
    plot_settings,
    display,
    kwargs...
)
    x_dim = BOLFI.x_dim(bolfi)
    bounds = bolfi.problem.domain.bounds
    count = size(samples)[2]
    X = bolfi.problem.data.X

    grid_size = plot_settings.grid_size
    steps = plot_steps(grid_size, bounds)

    res = x_dim * plot_settings.plot_cell_res
    fig = Figure(;
        size = (res, res),
    )
    limits = [(bounds[1][i], bounds[2][i]) for i in 1:x_dim]
    labels = isnothing(plot_settings.param_labels) ? ["x$i" for i in 1:x_dim] : plot_settings.param_labels
    @assert length(labels) == x_dim

    # single parameter marginals (diagonal)
    for dim in 1:x_dim
        xs = range(bounds[1][dim], bounds[2][dim]; length=grid_size) |> collect
        ys = zeros(length(xs))
        
        k = with_lengthscale(kernel, lengthscale[dim])
        for i in eachindex(xs)
            ys[i] = mean(k.(Ref(xs[i]), samples[dim,:]))
        end
        normalize_prob_vals!(ys, steps[dim])

        ax = Axis(fig[dim,dim]; xlabel=labels[dim], limits=(limits[dim],nothing))
        lines!(ax, xs, ys)
        plot_settings.plot_samples && scatter!(ax, samples[dim,:], zeros(count);
            SAMPLES_SCATTER_KWARGS...
        )
        plot_settings.plot_data && scatter!(ax, X[dim,:], zeros(size(X)[2]);
            DATA_SCATTER_KWARGS...
        )
    end

    # pair marginals
    for dim_a in 1:x_dim
        for dim_b in dim_a+1:x_dim
            xs_a = range(bounds[1][dim_a], bounds[2][dim_a]; length=grid_size)
            xs_b = range(bounds[1][dim_b], bounds[2][dim_b]; length=grid_size)
            xs = Iterators.product(xs_a, xs_b) |> collect .|> (t -> [t...])
            ys = zeros(size(xs))
            
            k = with_lengthscale(kernel, lengthscale[[dim_a,dim_b]])
            for i in eachindex(xs)
                ys[i] = mean(k.(Ref(xs[i]), eachcol(samples[[dim_a,dim_b],:])))
            end
            normalize_prob_vals!(ys, steps[dim_a], steps[dim_b])

            ax = Axis(fig[dim_b, dim_a]; xlabel=labels[dim_a], ylabel=labels[dim_b], limits=(limits[dim_a],limits[dim_b]))
            contourf!(ax, xs_a, xs_b, ys)
            plot_settings.plot_samples && scatter!(ax, samples[dim_a,:], samples[dim_b,:];
                SAMPLES_SCATTER_KWARGS...
            )
            plot_settings.plot_data && scatter!(ax, X[dim_a,:], X[dim_b,:];
                DATA_SCATTER_KWARGS...
            )
            if plot_settings.full_matrix
                ax_t = Axis(fig[dim_a, dim_b]; xlabel=labels[dim_b], ylabel=labels[dim_a], limits=(limits[dim_b],limits[dim_a]))
                contourf!(ax_t, xs_b, xs_a, ys')
                plot_settings.plot_samples && scatter!(ax_t, samples[dim_b,:], samples[dim_a,:];
                    SAMPLES_SCATTER_KWARGS...
                )
                plot_settings.plot_data && scatter!(ax_t, X[dim_b,:], X[dim_a,:];
                    DATA_SCATTER_KWARGS...
                )
            end
        end
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
