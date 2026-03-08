using BOSS
using CairoMakie
using Distributions

function plot_state(
    problem::BossProblem;
    f_true::Function,
    z_obs::AbstractVector{<:Real},
    n_points::Int = 500,
)
    @assert x_dim(problem) == 1 "plot_state only supports x_dim=1"
    @assert y_dim(problem) == 1 "plot_state only supports y_dim=1"
    @assert length(z_obs) == 1 "plot_state currently supports scalar observations"

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

    fig = Figure(size=(800, 1000))
    ax_func = Axis(
        fig[1, 1],
        title="Simulator",
        xlabel="Parameters x",
        ylabel="y = sim(x)",
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    ax_mse = Axis(
        fig[2, 1],
        title="Mean Squared Error",
        xlabel="Parameters x",
        ylabel="MSE",
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28,
        xticklabelsize=22,
        yticklabelsize=22,
    )

    hlines!(ax_func, [obs_mean], color=:orange, linewidth=2, label="observation")
    lines!(ax_func, xs, ys_true, color=:black, linewidth=3, linestyle=:dash, label="simulator function")
    scatter!(ax_func, X_data, [f_true([x])[1] for x in X_data], color=:red, markersize=14, label="evaluations")
    ylims!(ax_func, y_min_upper, y_max_upper)
    axislegend(ax_func, position=:lt, labelsize=22)

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
    axislegend(ax_mse, position=:lt, labelsize=22)

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
        fig_path = joinpath(cb.plot_dir, "iter_$(cb.iter).png")
        save(fig_path, fig)
        options.info && @info "Saved plot to $fig_path"
    end

    return nothing
end
