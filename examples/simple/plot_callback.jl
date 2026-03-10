using CairoMakie
using Distributions

mutable struct SimplePlotCallback <: BOSIP.BosipCallback
    iter::Int
    plot_dir::String
    plot_name::String
    step::Float64
    display::Bool
    save_plots::Bool
end

function SimplePlotCallback(; 
    plot_dir::String = "./examples/simple/plots",
    plot_name::String = "bosip_state",
    step::Float64 = 0.1,
    display::Bool = true,
    save_plots::Bool = true,
)
    return SimplePlotCallback(0, plot_dir, plot_name, step, display, save_plots)
end

function (cb::SimplePlotCallback)(bosip::BosipProblem; first::Bool, options::BOSS.BossOptions, kwargs...)
    if first
        cb.iter = 0
        _init_plot_dir(cb)
    else
        cb.iter += 1
    end

    fig = _plot_bosip_state(bosip; step=cb.step)

    cb.display && display(fig)
    if cb.save_plots
        file = joinpath(cb.plot_dir, "$(cb.plot_name)_$(lpad(cb.iter, 3, '0')).png")
        save(file, fig)
        options.info && @info "Saved plot: $file"
    end

    return nothing
end

function _init_plot_dir(cb::SimplePlotCallback)
    if cb.save_plots
        if isdir(cb.plot_dir)
            rm(cb.plot_dir; recursive=true)
        end
        mkpath(cb.plot_dir)
    end
    return nothing
end

function _plot_bosip_state(bosip::BosipProblem; step::Float64)
    @assert length(bounds[1]) == 2
    @assert length(bounds[2]) == 2

    xs = collect(bounds[1][1]:step:bounds[2][1])
    ys = collect(bounds[1][2]:step:bounds[2][2])

    true_post = _true_posterior()
    est_post = posterior_mean(bosip)
    post_var = posterior_variance(bosip)
    model_post = BOSS.model_posterior(bosip.problem)
    z_target = likelihood.z_obs[1]
    obs_noise_std = likelihood.std_obs[1]

    true_vals = _eval_grid(true_post, xs, ys)
    est_vals = _eval_grid(est_post, xs, ys)
    var_vals = _eval_grid(post_var, xs, ys)
    gp_mean_vals = _eval_grid(x -> _scalar_output(mean(model_post, x)), xs, ys)

    fig = Figure(size=(1200, 1000))

    ax11 = Axis(fig[1, 1], title="True posterior")
    contourf!(ax11, xs, ys, true_vals)
    _scatter_data!(ax11, bosip)

    ax12 = Axis(fig[1, 2], title="Estimated posterior mean")
    contourf!(ax12, xs, ys, est_vals)
    _scatter_data!(ax12, bosip)

    ax21 = Axis(fig[2, 1], title="Surrogate mean prediction")
    contourf!(ax21, xs, ys, gp_mean_vals)
    contour!(ax21, xs, ys, gp_mean_vals; levels=[z_target], color=:red, linewidth=2)
    contour!(
        ax21,
        xs,
        ys,
        gp_mean_vals;
        levels=[z_target - 2 * obs_noise_std, z_target + 2 * obs_noise_std],
        color=:red,
        linestyle=:dash,
        linewidth=2,
    )
    _scatter_data!(ax21, bosip)

    ax22 = Axis(fig[2, 2], title="Posterior variance")
    contourf!(ax22, xs, ys, var_vals)
    _scatter_data!(ax22, bosip)

    return fig
end

function _true_posterior()
    function post(x::AbstractVector{<:Real})
        return pdf(x_prior, x) * like(likelihood, f(x))
    end
    return post
end

function _eval_grid(func::Function, xs::AbstractVector, ys::AbstractVector)
    vals = Matrix{Float64}(undef, length(xs), length(ys))
    for i in eachindex(xs), j in eachindex(ys)
        vals[i, j] = _scalar_output(func([xs[i], ys[j]]))
    end
    return vals
end

_scalar_output(v::Real) = Float64(v)
_scalar_output(v::AbstractVector{<:Real}) = Float64(v[1])

function _scatter_data!(ax, bosip::BosipProblem)
    X = bosip.problem.data.X
    scatter!(ax, X[1, :], X[2, :]; color=:white, strokecolor=:black, strokewidth=1.5, markersize=10)
    return nothing
end
