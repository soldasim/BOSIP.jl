using BOSS
using CairoMakie
using Distributions

function set_theme_fonts!(; base_fontsize=20)
    set_theme!(
        fontsize = base_fontsize,
        Axis = (
            titlesize = base_fontsize + 2,
            xlabelsize = base_fontsize,
            ylabelsize = base_fontsize,
            xticklabelsize = base_fontsize - 2,
            yticklabelsize = base_fontsize - 2,
        ),
        Legend = (
            titlesize = base_fontsize,
            labelsize = base_fontsize - 1,
        ),
        Label = (
            textsize = base_fontsize,
        )
    )
end

function plot_state(
    problem::BossProblem;
    f_true::Function,
    z_obs::AbstractVector{<:Real},
    n_points::Int = 500,
    show_legend::Bool = false,
)
    @assert x_dim(problem) == 1 "plot_state only supports x_dim=1"
    @assert y_dim(problem) == 1 "plot_state only supports y_dim=1"
    @assert length(z_obs) == 1 "plot_state currently supports scalar observations"

    set_theme_fonts!();

    bounds = problem.domain.bounds
    x_min, x_max = bounds[1][1], bounds[2][1]
    xs = collect(range(x_min, x_max, length=n_points))

    gp_post = BOSS.model_posterior(problem)

    obs_mean = z_obs[1]

    ys_true = [f_true([x])[1] for x in xs]
    mses_true = [sum((f_true([x]) .- z_obs) .^ 2) / length(z_obs) for x in xs]

    mses_surrogate = [mean(gp_post, [x])[1] for x in xs]
    mses_surrogate_std = [std(gp_post, [x])[1] for x in xs]
    mses_surrogate_lower = mses_surrogate .- (2 .* mses_surrogate_std)
    mses_surrogate_upper = mses_surrogate .+ (2 .* mses_surrogate_std)

    X_data = vec(problem.data.X)
    Y_data = vec(problem.data.Y)

    y_min_upper = minimum(ys_true)
    y_max_upper = maximum(ys_true)
    y_margin = 0.05 * (y_max_upper - y_min_upper)
    y_min_upper -= y_margin
    y_max_upper += y_margin

    mse_min = minimum(mses_true)
    mse_max = maximum(mses_true)
    mse_margin = 0.05 * (mse_max - mse_min + eps())

    axis_height = 600
    fig = Figure(size=(400, axis_height))
    ax_func = Axis(
        fig[1, 1],
        title="Simulator",
        xlabel="Parameters x",
        ylabel="y = sim(x)",
    )
    ax_mse = Axis(
        fig[2, 1],
        title="Mean Squared Error",
        xlabel="Parameters x",
        ylabel="MSE",
    )

    hlines!(ax_func, [obs_mean], color=:orange, linewidth=2, label="observation")
    lines!(ax_func, xs, ys_true, color=:black, linewidth=3, linestyle=:dash, label="simulator function")
    scatter!(ax_func, X_data, [f_true([x])[1] for x in X_data], color=:red, markersize=14, label="evaluations")
    ylims!(ax_func, y_min_upper, y_max_upper)
    if show_legend
        axislegend(ax_func, position=:lt)
    end

    lines!(ax_mse, xs, mses_true, color=:black, linewidth=3, linestyle=:dash, label="true MSE")
    band!(ax_mse, xs, mses_surrogate_lower, mses_surrogate_upper, color=(:blue, 0.2))
    lines!(ax_mse, xs, mses_surrogate, color=:blue, linewidth=3, label="approx. MSE")
    scatter!(ax_mse, X_data, Y_data, color=:red, markersize=14, label="evaluations")

    best = result(problem)
    if !isnothing(best)
        best_x, best_y = best
        scatter!(ax_mse, [best_x[1]], [best_y[1]], color=:green, markersize=28, marker=:star5, label="optimum")
    end

    ylims!(ax_mse, mse_min - mse_margin, mse_max + mse_margin)
    if show_legend
        axislegend(ax_mse, position=:lt)
    end

    return fig
end

function plot_legend(; axis_height=600)
    set_theme_fonts!()
    
    fig = Figure(size=(600, 120), bgcolor=:white)
    ax = Axis(fig[1, 1], backgroundcolor=:white)
    
    # Create dummy plot elements with correct styling and labels (outside visible limits)
    lines!(ax, [-10, -9], [-10, -10], color=:black, linewidth=3, linestyle=:dash, label="simulator function")
    scatter!(ax, [-10], [-10], color=:red, markersize=14, label="data points")
    hlines!(ax, [-10], color=:orange, linewidth=2, label="observation")
    lines!(ax, [-10, -9], [-10, -10], color=:black, linewidth=3, linestyle=:dash, label="true MSE")
    lines!(ax, [-10, -9], [-10, -10], color=:blue, linewidth=3, label="surrogate model")
    scatter!(ax, [-10], [-10], color=:green, markersize=28, marker=:star5, label="optimum")
    
    # Create legend with horizontal layout
	axislegend(ax, position=:cc, orientation=:horizontal, nbanks=3, framevisible=true, backgroundcolor=(:white, 0.9))
    # Hide the axis but keep white background
    hidedecorations!(ax)
    hidespines!(ax)
    ax.limits = (0, 1, 0, 1)
    
    return fig
end

@kwdef mutable struct BossPlotCallback{F<:Function,V<:AbstractVector{<:Real}} <: BossCallback
    f_true::F
    z_obs::V
    iter::Int = 0
    display_plots::Bool = true
    plot_dir::String = "./examples/1d/plots_boss"
    save_plots::Bool = true
    n_points::Int = 500
end

function (cb::BossPlotCallback)(problem::BossProblem; first::Bool, options::BossOptions, kwargs...)
    cb.iter += 1

    if cb.save_plots && first
        isdir(cb.plot_dir) && rm(cb.plot_dir; recursive=true)
		mkpath(cb.plot_dir)
	end

    fig = plot_state(
        problem;
        f_true=cb.f_true,
        z_obs=cb.z_obs,
        n_points=cb.n_points,
    )

    if cb.display_plots
        display(fig)
    end

    if cb.save_plots
        fig_path = joinpath(cb.plot_dir, "boss_iter_$(cb.iter).pdf")
        save(fig_path, fig)
        options.info && @info "Saved plot to $fig_path"
    end

    return nothing
end
