
function plot_state_1d(bolfi, new_datum; plt, kwargs...)
    boss = bolfi.problem
    acquisition = unwrap(bolfi.problem.acquisition)

    post_mean = posterior_mean(bolfi)
    post_var = posterior_variance(bolfi)
    gp_post = BOSS.model_posterior(boss)

    acq_samples = 4
    acqs = [acquisition(bolfi, BolfiOptions()) for _ in 1:acq_samples]
    
    # --- PLOTS ---
    f = Figure(;
        size = (1440, 810),
    )

    ax = Axis(f[1,1];
        title = "true posterior",
    )
    plot_func_1d!(ax, x -> true_post([x]), bolfi, new_datum; plt)
    plot_data_1d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[1,2];
        title = "posterior mean",
    )
    plot_func_1d!(ax, x -> post_mean([x]), bolfi, new_datum; plt)
    plot_data_1d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[2,1];
        title = "abs. val. of GP mean",
    )
    plot_func_1d!(ax, x -> abs(mean(gp_post, [x])[1]), bolfi, new_datum; plt)
    plot_data_1d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[2,2];
        title = "acquisition",
    )
    plot_func_1d!(ax, x -> post_var([x]), bolfi, new_datum; plt, label="posterior variance", grid=plt.acq_grid, normalize=true, linewidth=3, linestyle=:dash)
    for a in acqs
        plot_func_1d!(ax, x -> a([x]), bolfi, new_datum; plt, normalize=true, grid=plt.acq_grid)
    end
    plot_data_1d!(ax, bolfi, new_datum; plt)
    axislegend(ax; position = :rt)

    return f
end

function plot_func_1d!(ax, func, bolfi, new_datum; plt, label=nothing, grid=nothing, normalize=false, kwargs...)
    xs = isnothing(grid) ? get_xs_1d(bolfi; plt) : grid
    
    if plt.parallel
        vals = calculate_values(func, xs)
    else
        vals = func.(xs)
    end
    
    normalize && normalize_values!(vals)
    lines!(ax, xs, vals;
        label,
        kwargs...
    )
end

function plot_data_1d!(ax, bolfi, new_datum; plt)
    boss = bolfi.problem

    scatter!(ax, vec(boss.data.X), zeros(length(boss.data.X));
        label = "data",
    )
    isnothing(new_datum) || scatter!(ax, new_datum, [0.];
        label = "new datum",
    )
end

function get_xs_1d(bolfi; plt)
    lb, ub = first.(bolfi.problem.domain.bounds)
    return lb:plt.step:ub
end
