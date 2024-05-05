using Plots
using Printf

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            rm(plot_dir, recursive=true)
        end
        mkdir(plot_dir)
    end
end


# - - - Plotting Callback - - - - -

mutable struct PlotCallback<: BolfiCallback
    iters::Int
    plot_each::Int
    term_cond::Union{TermCond, BolfiTermCond}
    save_plots::Bool
    plot_dir::String
    put_in_scale::Bool
end
PlotCallback(;
    plot_each = 1,
    term_cond,
    save_plots,
    plot_dir = ".",
    put_in_scale,
) = PlotCallback(0, plot_each, term_cond, save_plots, plot_dir, put_in_scale)

function (plt::PlotCallback)(problem::BolfiProblem; acquisition, options, kwargs...)
    plt.iters += 1
    if plt.iters % plt.plot_each == 0
        options.info && @info "Plotting ..."
        plot_state(problem; term_cond=plt.term_cond, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name="p_$(plt.iters)", put_in_scale=plt.put_in_scale, noise_vars_true=ToyProblem.σe_true.^2, acquisition=acquisition.acq)
    end
end


# - - - Plotting Scripts - - - - -

function separate_new_datum(problem)
    bolfi = deepcopy(problem)
    new = bolfi.problem.data.X[:,end]
    bolfi.problem.data.X = bolfi.problem.data.X[:, 1:end-1]
    bolfi.problem.data.Y = bolfi.problem.data.Y[:, 1:end-1]
    return bolfi, new
end

function plot_state(problem; term_cond, display=true, save_plots=false, plot_dir=".", plot_name="p", put_in_scale=false, noise_vars_true, acquisition)
    bolfi, new_datum = separate_new_datum(problem)
    p = plot_sets(bolfi; new_datum, term_cond, display, put_in_scale, noise_vars_true, acquisition)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_sets(bolfi; new_datum=nothing, term_cond, display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05)
    @assert acquisition isa SetsPostVariance
    subset_plots = [plot_samples(get_subset(bolfi, set); new_datum, term_cond, display=false, put_in_scale, noise_vars_true=noise_vars_true[set], acquisition=PostVariance(), step, y_set=set) for set in eachcol(bolfi.y_sets)]
    acq_plot = plot_acquisition(bolfi; new_datum, acquisition, step)
    p = plot(subset_plots..., acq_plot; layout=(length(subset_plots)+1, 1), size=(1440, (length(subset_plots)+1)*810))
    display && Plots.display(p)
    return p
end

function plot_acquisition(bolfi; new_datum, acquisition, step=0.05)
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
    plot_samples!(p4, X; new_datum, label=nothing)
end

function plot_samples(bolfi; new_datum=nothing, term_cond, display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05, y_set=fill(true, ToyProblem.y_dim), title=nothing)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    # Confidence
    if all(hasproperty.(Ref(term_cond), [:n, :q]))
        n = term_cond.n
        q = term_cond.q
        @info "Plotting with `q = $q` and `n = $n`."
    else
        n = 1.  # num of std from GP mean
        q = 0.8  # confidence of the confidence set
        @warn "Plotting with hard-coded `q = $q` and `n = $n`."
    end

    # unnormalized posterior likelihood `p(y | x) * p(x) ∝ p(x | y)`
    function ll_post(x)
        y = ToyProblem.experiment(x; noise_vars=zeros(ToyProblem.y_dim))[y_set]
        
        # ps = numerical_issues(x) ? 0. : 1.
        isnothing(y) && return 0.

        ll = pdf(MvNormal(y, sqrt.(noise_vars_true)), ToyProblem.y_obs[y_set])
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # confidence sets
    xs = rand(x_prior, 10_000)

    post_real, c_real, V_real = find_cutoff(ll_post, x_prior, q; xs)  # unnormalized
    post_μ, c_μ, V_μ = find_cutoff(gp_post, bolfi.var_e, x_prior, q; xs, normalize=true)
    # post_med, c_med, V_med = find_cutoff(gp_mean(gp_post), bolfi.var_e, x_prior, q; xs, normalize=true)
    post_lb, c_lb, V_lb = find_cutoff(gp_bound(gp_post, -n), bolfi.var_e, x_prior, q; xs, normalize=true)
    post_ub, c_ub, V_ub = find_cutoff(gp_bound(gp_post, +n), bolfi.var_e, x_prior, q; xs, normalize=true)
    # post_max, c_max, V_max = find_cutoff_max(bolfi, q, n; xs, normalize=true)

    conf_sets_real = [
        (p=post_real, c=c_real, V=V_real, label="real q:$q ($(@sprintf("%.4f", V_real)))", color=:greenyellow),
        (p=post_μ, c=c_μ, V=V_μ, label="mean q:$q ($(@sprintf("%.4f", V_μ)))", color=:cyan),
    ]
    conf_sets_approx = [
        (p=post_μ, c=c_μ, V=V_μ, label="mean q:$q ($(@sprintf("%.4f", V_μ)))", color=:cyan),
        # (p=post_med, c=c_med, V=V_med, label="median q:$q ($(@sprintf("%.4f", V_med)))", color=:teal),
        (p=post_lb, c=c_lb, V=V_lb, label="GP-LB q:$q ($(@sprintf("%.4f", V_lb)))", color=:yellow, linestyle=:dash),
        (p=post_ub, c=c_ub, V=V_ub, label="GP-UB q:$q ($(@sprintf("%.4f", V_ub)))", color=:yellow, linestyle=:dashdot),
        # (p=post_max, c=c_max, V=V_max, label="max q:$q ($(@sprintf("%.4f", V_max)))", color=:maroon),
    ]

    in_real = (post_real.(eachcol(xs)) .> c_real)
    in_mean = (post_μ.(eachcol(xs)) .> c_μ)
    mean_real_ratio = set_overlap(in_real, in_mean, x_prior, xs)
    
    # gp-approximated posterior likelihood
    ll_gp(a, b) = post_μ([a, b])

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())
    acq_name = split(string(typeof(acquisition)), '.')[end]

    # - - - PLOT - - - - -
    clims = nothing
    if put_in_scale
        _, max_ll = maximize_prima(ll_post, x_prior, bounds; multistart=32, rhoend=1e-4)
        clims = (0., 1.2*max_ll)
    end
    kwargs = (colorbar=false,)

    p1 = plot(; title="true posterior", clims, kwargs...)
    plot_posterior!(p1, (a,b)->ll_post([a, b]); conf_sets=conf_sets_real, lims, label=nothing, step)
    plot_samples!(p1, X; new_datum, label=nothing)
    scatter!(p1, [], []; label="mean-real ratio = $(@sprintf("%.4f", mean_real_ratio))", color=nothing)

    p2 = plot(; title="approx. posterior", clims, kwargs...)
    plot_posterior!(p2, ll_gp; conf_sets=conf_sets_approx, lims, label=nothing, step)
    plot_samples!(p2, X; new_datum, label=nothing)
    # scatter!(p2, [], []; label="med/mean = $(@sprintf("%.4f", V_med / V_μ))", color=nothing)
    # scatter!(p2, [], []; label="mean/max = $(@sprintf("%.4f", V_μ / V_max))", color=nothing)
    # TODO uncomment below
    # scatter!(p2, [], []; label="lbub ratio = $(@sprintf("%.4f", BOLFI.calculate(term_cond, bolfi)))", color=nothing)

    p3 = plot(; title="abs(GP[1] mean)", kwargs...)
    plot_posterior!(p3, (a,b) -> abs(gp_post([a,b])[1][1]); lims, label=nothing, step)
    plot_samples!(p3, X; new_datum, label=nothing)

    p4 = plot(; title="acquisition " * acq_name, kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); lims, label=nothing, step)
    plot_samples!(p4, X; new_datum, label=nothing)

    isnothing(title) && (title = put_in_scale ? "(in scale)" : "(not in scale)")
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; conf_sets=[], lims, label=nothing, step=0.05)
    grid = lims[1]:step:lims[2]
    vals = ll.(grid', grid)
    clims = (minimum(vals), maximum(vals))
    contourf!(p, grid, grid, vals; clims)

    # "OBSERVATION-RULES"
    obs_color = :white
    plot!(p, a->1/a, grid; y_lims=lims, label, color=obs_color)
    plot!(p, a->a, grid; y_lims=lims, label, color=obs_color)

    # CONFIDENCE SET
    target = mean(vals)
    for (f, c, V, kwargs...) in conf_sets
        plot_confidence_set!(p, f, c; target, lims, step, V, linewidth=2, kwargs...)
    end
    return p
end

function plot_samples!(p, samples; new_datum=nothing, label="(a,b) ~ p(a,b|d)")
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
end

function plot_confidence_set!(p, ll, c; target, lims, step, label=nothing, V=nothing, color=:red, kwargs...)
    grid = lims[1]:step:lims[2]
    norm = target / c  # s.t. the contour will be within climss
    contour!(p, grid, grid, (a,b)->norm*ll([a,b]); levels=[norm*c], color, kwargs...)
    isnothing(label) || scatter!(p, [], []; label, color)
end
