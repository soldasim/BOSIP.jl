
isfalse(x) = (x == false)

function plot_state_2d(bolfi, new_datum;
    plt,
    sample_reference = true, # boolean or a matrix of samples
    sample_posterior = true, # boolean or a matrix of samples
    kwargs...
)
    boss = bolfi.problem
    acquisition = unwrap(bolfi.problem.acquisition)

    logpost_est = log_posterior_estimate()
    approx_post = approx_posterior(bolfi)
    post_mean = posterior_mean(bolfi)
    log_approx_post = log_approx_posterior(bolfi)
    log_post_mean = log_posterior_mean(bolfi)
    post_var = posterior_variance(bolfi)
    gp_post = BOSS.model_posterior(boss)
    like = approx_likelihood(bolfi)
    loglike = log_approx_likelihood(bolfi)

    acq = acquisition(bolfi, BolfiOptions())

    # --- PLOTS ---
    f = Figure(;
        # size = (1440, 810),
        size = (1440, 1620),
    )

    ax = Axis(f[1,1];
        title = "true posterior",
    )
    plot_func_2d!(ax, (x,y) -> exp(ToyProblem.true_logpost([x,y])), bolfi, new_datum; plt)
    isfalse(sample_reference) || plot_posterior_samples!(ax, ToyProblem.true_logpost, plt.sampler, sample_reference, bolfi)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[1,2];
        title = "posterior mean",
    )
    plot_func_2d!(ax, (x,y) -> post_mean([x,y]), bolfi, new_datum; plt)
    isfalse(sample_posterior) || plot_posterior_samples!(ax, log_post_mean, plt.sampler, sample_posterior, bolfi)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[2,1];
        title = "abs. val. of GP mean",
    )
    plot_func_2d!(ax, (x,y) -> abs(mean(gp_post, [x,y])[1]), bolfi, new_datum; plt)
    plot_data_2d!(ax, bolfi, new_datum; plt)

    ax = Axis(f[2,2];
        title = "posterior variance",
    )
    plot_func_2d!(ax, (x,y) -> post_var([x,y]), bolfi, new_datum; plt, grid=plt.acq_grid)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    return f
end

function plot_func_2d!(ax, func, bolfi, new_datum; plt, step=plt.step, label=nothing, grid=nothing, normalize=false, log_scale=false, kwargs...)
    xs, ys = isnothing(grid) ? get_xs_2d(bolfi; step) : (grid, grid)
    
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

function plot_data_2d!(ax, bolfi, new_datum; plt)
    boss = bolfi.problem

    scatter!(ax, boss.data.X[1,:], boss.data.X[2,:];
        label = "data",
        color = :cyan,
    )
    isnothing(new_datum) || scatter!(ax, [new_datum[1]], [new_datum[2]];
        label = "new datum",
        color = :orange,
    )
end

function plot_posterior_samples!(ax, logpost, sampler, samples, bolfi)
    domain = bolfi.problem.domain
    bounds = domain.bounds

    if samples isa Bool
        @assert samples
        xs = BOLFI.pure_sample_posterior(sampler, logpost, domain, 1000)
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

function get_xs_2d(bolfi; step)
    lb1, ub1 = first.(bolfi.problem.domain.bounds)
    lb2, ub2 = getindex.(bolfi.problem.domain.bounds, Ref(2))
    return lb1:step:ub1, lb2:step:ub2
end
