using Plots

function plot_state(problem; display=true, save_plots=false, iters, put_in_scale, noise_vars_true, acquisition)
    p = plot_sets(problem; display, put_in_scale, noise_vars_true, acquisition)
    save_plots && savefig(p, "p_$(iters).png")
end

function plot_sets(bolfi; display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05)
    @assert acquisition isa SetsPostVariance
    subset_plots = [plot_samples(get_subset(bolfi, set); display=false, put_in_scale, noise_vars_true=noise_vars_true[set], acquisition=PostVariance(), step, y_set=set) for set in eachcol(bolfi.y_sets)]
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

    acq = acquisition(bolfi, BOSS.BossOptions())
    acq_name = split(string(typeof(acquisition)), '.')[end]

    p4 = plot(; title="acquisition " * acq_name, colorbar=false)
    plot_posterior!(p4, (a,b) -> acq([a,b]); lims, label=nothing, step)
    plot_samples!(p4, X; label=nothing)
end

function plot_samples(bolfi; display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05, y_set=fill(true, ToyProblem.y_dim), title=nothing)
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
    post_μ = BOLFI.posterior_mean(x_prior, gp_post; var_e=bolfi.var_e)
    ll_gp(a, b) = post_μ([a, b])

    # acquisition
    acq = acquisition(bolfi, BOSS.BossOptions())
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

    p2 = plot(; title="(unnormalized) approx. posterior", clims, kwargs...)
    plot_posterior!(p2, ll_gp; lims, label=nothing, step)
    plot_samples!(p2, X; label=nothing)

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

function plot_posterior!(p, ll; lims, label="ab=d", step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, ll)
    
    # "OBSERVATION-RULES"
    obs_color = :gold
    plot!(p, a->1/a, grid; y_lims=lims, label, color=obs_color)
    plot!(p, a->0., grid; y_lims=lims, label, color=obs_color)
end

function plot_samples!(p, samples; label="(a,b) ~ p(a,b|d)")
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
end
