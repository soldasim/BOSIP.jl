module CairoExt

using BOLFI, BOSS
using CairoMakie
using BOLFI.KernelFunctions
using BOLFI.Statistics
using BOLFI.LinearAlgebra
using BOLFI.Distributions

@kwdef struct PlotSettings <: BOLFI.PlotSettings
    plot_step::Float64 = 0.1
    param_labels::Union{Nothing, Vector{String}} = nothing
end

BOLFI.PlotSettings(args...; kwargs...) =
    PlotSettings(args...; kwargs...)

function BOLFI.plot_marginals_int(bolfi::BolfiProblem;
    grid_size = 1000,
    plot_settings = PlotSettings(),
    info = true,
    display = true,
)
    info && @info "Generating a latin hypercube of parameter samples ..."
    xs = BOSS.generate_LHC(bolfi.problem.domain.bounds, grid_size)
    keep = BOSS.in_domain.(eachcol(xs), Ref(bolfi.problem.domain))
    info && (sum(keep) < grid_size) && @warn "Discarding some LHC points outside of the parameter domain."
    xs = xs[:, keep]

    info && @info "Plotting the marginals ..."
    return plot_marginals_int(bolfi, xs; plot_settings, display)
end

function BOLFI.plot_marginals_kde(bolfi::BolfiProblem;
    turing_options = TuringOptions(),
    kernel = GaussianKernel(),
    plot_settings = PlotSettings(),
    info = true,
    display = true,
)
    info && @info "Sampling parameter samples from the posterior ..."
    xs = sample_posterior(bolfi, turing_options)

    info && @info "Plotting the marginals ..."
    return plot_marginals_kde(bolfi, xs, kernel; plot_settings, display)
end

function plot_marginals_int(bolfi::BolfiProblem, grid::AbstractMatrix{<:Real};
    plot_settings,
    display,
    kwargs...    
)
    x_dim = BOLFI.x_dim(bolfi)
    count = size(grid)[2]
    bounds = bolfi.problem.domain.bounds
    approx_post = approx_posterior(bolfi; normalize=true, xs=grid)

    fig = Figure()
    labels = isnothing(plot_settings.param_labels) ? ["x$i" for i in 1:x_dim] : plot_settings.param_labels
    @assert length(labels) == x_dim

    tmp_row_a = zeros(count)
    tmp_row_b = zeros(count)

    # single parameter marginals (diagonal)
    for dim in 1:x_dim
        tmp_row_a .= grid[dim,:]

        xs = bounds[1][dim]:plot_settings.plot_step:bounds[2][dim] |> collect
        ys = zeros(length(xs))
        
        for i in eachindex(xs)
            grid[dim,:] .= xs[i]
            ys[i] = sum(approx_post.(eachcol(grid)))
        end

        ax = Axis(fig[dim,dim]; xlabel=labels[dim])
        lines!(ax, xs, ys)

        grid[dim,:] .= tmp_row_a
    end

    # pair marginals
    for dim_a in 1:x_dim
        for dim_b in dim_a+1:x_dim
            tmp_row_a .= grid[dim_a,:]
            tmp_row_b .= grid[dim_b,:]

            xs_a = bounds[1][dim_a]:plot_settings.plot_step:bounds[2][dim_a]
            xs_b = bounds[1][dim_b]:plot_settings.plot_step:bounds[2][dim_b]
            xs = Iterators.product(xs_a, xs_b) |> collect
            ys = zeros(size(xs))
            
            for i in eachindex(xs)
                x_ = xs[i]
                grid[dim_a,:] .= x_[1]
                grid[dim_b,:] .= x_[2]
                ys[i] = sum(approx_post.(eachcol(grid)))
            end

            ax = Axis(fig[dim_b, dim_a]; xlabel=labels[dim_a], ylabel=labels[dim_b])
            contourf!(ax, xs_a, xs_b, ys)
            ax_t = Axis(fig[dim_a, dim_b]; xlabel=labels[dim_b], ylabel=labels[dim_a])
            contourf!(ax_t, xs_b, xs_a, ys')

            grid[dim_a,:] .= tmp_row_a
            grid[dim_b,:] .= tmp_row_b
        end
    end

    display && CairoMakie.display(fig)
    return fig
end

function plot_marginals_kde(bolfi::BolfiProblem, samples::AbstractMatrix{<:Real}, kernel;
    plot_settings,
    display,
    kwargs...    
)
    x_dim = BOLFI.x_dim(bolfi)
    bounds = bolfi.problem.domain.bounds
    count = size(samples)[2]

    fig = Figure()
    limits = (bounds[1][1], bounds[2][1]), (bounds[1][2], bounds[2][2])
    labels = isnothing(plot_settings.param_labels) ? ["x$i" for i in 1:x_dim] : plot_settings.param_labels
    @assert length(labels) == x_dim

    # single parameter marginals (diagonal)
    for dim in 1:x_dim
        xs = bounds[1][dim]:plot_settings.plot_step:bounds[2][dim] |> collect
        ys = zeros(length(xs))
        
        for i in eachindex(xs)
            ys[i] = mean(kernel.(Ref(xs[i]), samples[dim,:]))
        end

        ax = Axis(fig[dim,dim]; xlabel=labels[dim])
        lines!(ax, xs, ys)
        scatter!(ax, samples[dim,:], zeros(count);
            color = :green,
            marker = :xcross,
            markersize = 3,
        )
    end

    # pair marginals
    for dim_a in 1:x_dim
        for dim_b in dim_a+1:x_dim
            xs_a = bounds[1][dim_a]:plot_settings.plot_step:bounds[2][dim_a]
            xs_b = bounds[1][dim_b]:plot_settings.plot_step:bounds[2][dim_b]
            xs = Iterators.product(xs_a, xs_b) |> collect .|> (t -> [t...])
            ys = zeros(size(xs))
            
            for i in eachindex(xs)
                ys[i] = mean(kernel.(Ref(xs[i]), eachcol(samples[[dim_a,dim_b],:])))
            end

            ax = Axis(fig[dim_b, dim_a]; xlabel=labels[dim_a], ylabel=labels[dim_b], limits)
            contourf!(ax, xs_a, xs_b, ys)
            scatter!(ax, samples[dim_a,:], samples[dim_b,:];
                color = :green,
                marker = :xcross,
                markersize = 3,
            )
            ax_t = Axis(fig[dim_a, dim_b]; xlabel=labels[dim_b], ylabel=labels[dim_a], limits)
            contourf!(ax_t, xs_b, xs_a, ys')
            scatter!(ax_t, samples[dim_b,:], samples[dim_a,:];
                color = :green,
                marker = :xcross,
                markersize = 3,
            )
        end
    end

    display && CairoMakie.display(fig)
    return fig
end

end # module CairoExt
