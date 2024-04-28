using Plots
using Printf


# - - - Plotting Callback - - - - -

mutable struct PlotCallback<: BolfiCallback
    iters::Int
    q::Float64
    save_plots::Bool
    put_in_scale::Bool
end
PlotCallback(;
    q,
    save_plots,
    put_in_scale,
) = PlotCallback(0, q, save_plots, put_in_scale)

function (plt::PlotCallback)(problem::BolfiProblem; acquisition, kwargs...)
    plt.iters += 1
    plot_state(problem; q=plt.q, save_plots=plt.save_plots, iters=plt.iters, put_in_scale=plt.put_in_scale, noise_vars_true=ToyProblem.σe_true.^2, acquisition=acquisition.acq)
end


# - - - Plotting Scripts - - - - -

function plot_state(problem; q, display=true, save_plots=false, iters, put_in_scale, noise_vars_true, acquisition)
    p = plot_sets(problem; q, display, put_in_scale, noise_vars_true, acquisition)
    save_plots && savefig(p, "p_$(iters).png")
end

function plot_sets(bolfi; q, display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05)
    @assert acquisition isa SetsPostVariance
    subset_plots = [plot_samples(get_subset(bolfi, set); q, display=false, put_in_scale, noise_vars_true=noise_vars_true[set], acquisition=PostVariance(), step, y_set=set) for set in eachcol(bolfi.y_sets)]
    acq_plot = plot_acquisition(bolfi; acquisition, step)
    p = plot(subset_plots..., acq_plot; layout=(length(subset_plots)+1, 1), size=(1440, (length(subset_plots)+1)*810))
    display && Plots.display(p)
    return p
end

function plot_acquisition(bolfi; acquisition, step=0.05)
    problem = bolfi.problem
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    acq = acquisition(bolfi, BolfiOptions())
    acq_name = split(string(typeof(acquisition)), '.')[end]

    p4 = plot(; title="acquisition " * acq_name, colorbar=false)
    plot_posterior!(p4, (a,b) -> acq([a,b]); lims, label=nothing, step)
    plot_samples!(p4, X; label=nothing)
end

function plot_samples(bolfi; q, display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05, y_set=fill(true, ToyProblem.y_dim), title=nothing)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood `p(d | a, b) * p(a, b) ∝ p(a, b | d)`
    function ll_post(a, b)
        x = [a, b]
        y = ToyProblem.experiment(x; noise_vars=zeros(ToyProblem.y_dim))[y_set]
        
        # ps = numerical_issues(x) ? 0. : 1.
        isnothing(y) && return 0.

        ll = pdf(MvNormal(y, sqrt.(noise_vars_true)), ToyProblem.y_obs[y_set])
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # gp-approximated posterior likelihood
    xs = rand(x_prior, 10_000)
    post_μ, c_μ, V_μ = find_cutoff(gp_post, x_prior, bolfi.var_e, q; xs, normalize=true)
    post_med, c_med, V_med = find_cutoff(gp_quantile(gp_post, 0.5), x_prior, bolfi.var_e, q; xs, normalize=true)
    conf_sets = [
        (post_μ, c_μ, V_μ, "expected q:$q ($(@sprintf("%.4f", V_μ)))", :red),
        (post_med, c_med, V_med, "median q:$q ($(@sprintf("%.4f", V_med)))", :white),
    ]
    
    ll_gp(a, b) = post_μ([a, b])

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())
    acq_name = split(string(typeof(acquisition)), '.')[end]

    # - - - PLOT - - - - -
    clims = nothing
    if put_in_scale
        _, max_ll = maximize_prima((x)->ll_post(x...), x_prior, bounds; multistart=32, rhoend=1e-4)
        clims = (0., 1.2*max_ll)
    end
    kwargs = (colorbar=false,)

    p1 = plot(; title="(unnormalized) true posterior", clims, kwargs...)
    plot_posterior!(p1, ll_post; lims, label=nothing, step)
    plot_samples!(p1, X; label=nothing)

    p2 = plot(; title="(normalized) approx. posterior", clims, kwargs...)
    plot_posterior!(p2, ll_gp; conf_sets, lims, label=nothing, step)
    plot_samples!(p2, X; label=nothing)
    scatter!(p2, [], []; label="med/exp = $(@sprintf("%.4f", V_med / V_μ))", color=nothing)

    p3 = plot(; title="GP[1] mean", kwargs...)
    plot_posterior!(p3, (a,b) -> gp_post([a,b])[1][1]; lims, label=nothing, step)
    plot_samples!(p3, X; label=nothing)

    p4 = plot(; title="acquisition " * acq_name, kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); lims, label=nothing, step)
    plot_samples!(p4, X; label=nothing)

    isnothing(title) && (title = put_in_scale ? "(in scale)" : "(not in scale)")
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; conf_sets=[], lims, label=nothing, step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, ll)
    
    # "OBSERVATION-RULES"
    obs_color = :gold
    plot!(p, a->1/a, grid; y_lims=lims, label, color=obs_color)
    plot!(p, a->0., grid; y_lims=lims, label, color=obs_color)

    # CONFIDENCE SET
    for (f, c, V, label, color) in conf_sets
        plot_confidence_set!(p, (a,b)->f([a,b]), c; lims, step, label, V, color)
    end
    return p
end

function plot_samples!(p, samples; label="(a,b) ~ p(a,b|d)")
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
end

function plot_confidence_set!(p, ll, c; lims, step, label=nothing, V=nothing, color=:red, kwargs...)
    grid = lims[1]:step:lims[2]
    contour!(p, grid, grid, ll; levels=[c], color, kwargs...)
    isnothing(label) || scatter!(p, [], []; label, color)
end
