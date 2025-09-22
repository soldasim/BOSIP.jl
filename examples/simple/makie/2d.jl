
isfalse(x) = (x == false)

function plot_state_2d(bosip, new_datum;
    plt,
    sample_reference = true, # boolean or a matrix of samples
    sample_posterior = true, # boolean or a matrix of samples
    kwargs...
)
    boss = bosip.problem
    acquisition = unwrap(bosip.problem.acquisition)

    logpost_est = log_posterior_estimate()
    approx_post = approx_posterior(bosip)
    post_mean = posterior_mean(bosip)
    log_approx_post = log_approx_posterior(bosip)
    log_post_mean = log_posterior_mean(bosip)
    post_var = posterior_variance(bosip)
    gp_post = BOSS.model_posterior(boss)
    like = approx_likelihood(bosip)
    loglike = log_approx_likelihood(bosip)

    # acq_samples = 2
    # acqs = [acquisition(bosip, BosipOptions()) for _ in 1:acq_samples]
    acq = acquisition(bosip, BosipOptions())

    # --- PLOTS ---
    f = Figure(;
        # size = (1440, 810),
        size = (1440, 1620),
    )

    ax = Axis(f[1,1];
        title = "true posterior",
    )
    plot_func_2d!(ax, (x,y) -> exp(ToyProblem.true_logpost([x,y])), bosip, new_datum; plt)
    isfalse(sample_reference) || plot_posterior_samples!(ax, ToyProblem.true_logpost, plt.sampler, sample_reference, bosip)
    plot_data_2d!(ax, bosip, new_datum; plt)
    # axislegend(ax; position = :rt)

    if logpost_est == log_posterior_mean
        ax = Axis(f[1,2];
            title = "posterior mean",
        )
        plot_func_2d!(ax, (x,y) -> post_mean([x,y]), bosip, new_datum; plt)
        isfalse(sample_posterior) || plot_posterior_samples!(ax, log_post_mean, plt.sampler, sample_posterior, bosip)
        plot_data_2d!(ax, bosip, new_datum; plt)
        # axislegend(ax; position = :rt)
    elseif logpost_est == log_approx_posterior
        ax = Axis(f[1,2];
            title = "median posterior",
        )
        plot_func_2d!(ax, (x,y) -> approx_post([x,y]), bosip, new_datum; plt)
        isfalse(sample_posterior) || plot_posterior_samples!(ax, log_approx_post, plt.sampler, sample_posterior, bosip)
        plot_data_2d!(ax, bosip, new_datum; plt)
        # axislegend(ax; position = :rt)
    else
        @assert false
    end

    # ax = Axis(f[2,1];
    #     title = "abs. val. of GP mean",
    # )
    # plot_func_2d!(ax, (x,y) -> abs(mean(gp_post, [x,y])[1]), bosip, new_datum; plt)
    # plot_data_2d!(ax, bosip, new_datum; plt)
    # # axislegend(ax; position = :rt)

    # ax = Axis(f[2,1];
    #     title = "acquisition",
    # )
    # plot_func_2d!(ax, (x,y) -> acq([x,y]), bosip, new_datum; plt, step=plt.step*10)
    # hasfield(typeof(acq), :xs) && scatter!(ax, acq.xs[1,:], acq.xs[2,:])
    # plot_data_2d!(ax, bosip, new_datum; plt)
    # axislegend(ax; position = :rt)
    ax = Axis(f[2,1];
        title = "median log-likelihood",
    )
    plot_func_2d!(ax, (x,y) -> loglike([x,y]), bosip, new_datum; plt, grid=plt.acq_grid)
    plot_data_2d!(ax, bosip, new_datum; plt)
    # axislegend(ax; position = :rt)

    # ax = Axis(f[2,2];
    #     title = "posterior variance",
    # )
    # plot_func_2d!(ax, (x,y) -> post_var([x,y]), bosip, new_datum; plt, grid=plt.acq_grid)
    # plot_data_2d!(ax, bosip, new_datum; plt)
    # # axislegend(ax; position = :rt)
    ax = Axis(f[2,2];
        title = "median likelihood",
    )
    plot_func_2d!(ax, (x,y) -> like([x,y]), bosip, new_datum; plt, grid=plt.acq_grid)
    plot_data_2d!(ax, bosip, new_datum; plt)
    # axislegend(ax; position = :rt)

    return f
end

function plot_func_2d!(ax, func, bosip, new_datum; plt, step=plt.step, label=nothing, grid=nothing, normalize=false, log_scale=false, kwargs...)
    xs, ys = isnothing(grid) ? get_xs_2d(bosip; step) : (grid, grid)
    
    if plt.parallel
        vals = calculate_values(t -> func(t...), Iterators.product(xs, ys))
    else
        vals = map(t -> func(t...), Iterators.product(xs, ys))
    end

    if log_scale
        M = minimum(vals)
        vals .+= M + 1.
        vals = log.(vals)
    end

    normalize && normalize_values!(vals)
    contourf!(ax, xs, ys, vals;
        label,
        kwargs...
    )
end

function plot_data_2d!(ax, bosip, new_datum; plt)
    boss = bosip.problem

    scatter!(ax, boss.data.X[1,:], boss.data.X[2,:];
        label = "data",
        color = :cyan,
    )
    isnothing(new_datum) || scatter!(ax, [new_datum[1]], [new_datum[2]];
        label = "new datum",
        color = :orange,
    )
end

function plot_posterior_samples!(ax, logpost, sampler, samples, bosip)
    domain = bosip.problem.domain
    bounds = domain.bounds

    if samples isa Bool
        @assert samples
        xs = BOSIP.sample_posterior_pure(sampler, logpost, domain, 1000)
    else
        xs = samples
    end

    # fix limits so that samples out of bounds are not plotted
    xlims = bounds[1][1], bounds[2][1]
    ylims = bounds[1][2], bounds[2][2]
    xlims!(ax, xlims)
    ylims!(ax, ylims)

    scatter!(ax, xs;
        color = :green,
        marker = :xcross,
        markersize = 6,
    )
end

function get_xs_2d(bosip; step)
    lb1, ub1 = first.(bosip.problem.domain.bounds)
    lb2, ub2 = getindex.(bosip.problem.domain.bounds, Ref(2))
    return lb1:step:ub1, lb2:step:ub2
end
