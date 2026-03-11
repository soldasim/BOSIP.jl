using BOSIP
using BOSS
using CairoMakie
using Distributions

function plot_state(p::BosipProblem; n_points::Int = 500, separate_figures::Bool = false, show_surrogate_in_simulator::Bool = true, credible_interval_level::Union{Nothing, Float64} = 0.95)
	@assert x_dim(p) == 1 "plot_state only supports x_dim=1"
	@assert y_dim(p) == 1 "plot_state only supports y_dim=1"

	bounds = p.problem.domain.bounds
	x_min, x_max = bounds[1][1], bounds[2][1]
	xs = collect(range(x_min, x_max, length=n_points))

	f_true = p.problem.f
	gp_post = BOSS.model_posterior(p.problem)
	approx_post = approx_posterior(p; normalize=false)
	post_var = posterior_variance(p; normalize=false)
	@assert hasproperty(p.likelihood, :z_obs) "plot_state requires likelihood with z_obs"
	@assert hasproperty(p.likelihood, :std_obs) "plot_state requires likelihood with std_obs"
	obs_mean = p.likelihood.z_obs[1]
	obs_std = p.likelihood.std_obs[1]
	obs_lower = obs_mean - 2 * obs_std
	obs_upper = obs_mean + 2 * obs_std

	_f_y(x) = let r = f_true(x); r isa Tuple ? r[1] : r end
	ys_true = [_f_y([x])[1] for x in xs]
	ys_surrogate = [mean(gp_post, [x])[1] for x in xs]
	ys_surrogate_std = [std(gp_post, [x])[1] for x in xs]
	ys_surrogate_lower = ys_surrogate .- (2 .* ys_surrogate_std)
	ys_surrogate_upper = ys_surrogate .+ (2 .* ys_surrogate_std)

	logpost_true = [Distributions.logpdf(p.x_prior, [x]) + loglike(p.likelihood, _f_y([x])) for x in xs]
	post_true = exp.(logpost_true)
	post_approx = [approx_post([x])[1] for x in xs]
	post_approx_std = [sqrt(max(post_var([x])[1], 0.0)) for x in xs]
	post_approx_lower = max.(post_approx .- (2 .* post_approx_std), 0.0)
	post_approx_upper = post_approx .+ (2 .* post_approx_std)
	# Calculate credible interval(s) using HPD approach for multimodal posteriors
	ci_intervals = Vector{Tuple{Float64, Float64}}()
	if !isnothing(credible_interval_level)
		# Normalize the approximate posterior to get a probability distribution
		dx = xs[2] - xs[1]  # grid spacing
		norm_const = sum(post_approx) * dx
		post_prob = post_approx ./ norm_const
		
		# Find cut-off threshold using HPD approach
		# Sort probability values in descending order
		sorted_probs = sort(post_prob, rev=true)
		cumsum_probs = cumsum(sorted_probs) .* dx
		
		# Find the cut-off probability where cumulative mass equals credible_interval_level
		cutoff_idx = findfirst(cumsum_probs .>= credible_interval_level)
		if !isnothing(cutoff_idx)
			cutoff_prob = sorted_probs[cutoff_idx]
			
			# Find all contiguous regions where post_prob > cutoff_prob
			in_region = post_prob .> cutoff_prob
			region_start = nothing
			for i in 1:length(in_region)
				if in_region[i] && isnothing(region_start)
					# Start of a new region
					region_start = i
				elseif !in_region[i] && !isnothing(region_start)
					# End of a region
					push!(ci_intervals, (xs[region_start], xs[i-1]))
					region_start = nothing
				end
			end
			# Close final region if it extends to the end
			if !isnothing(region_start)
				push!(ci_intervals, (xs[region_start], xs[end]))
			end
		end
	end

	X_data = vec(p.problem.data.X)
	Y_data = vec(p.problem.data.Y)

	y_min_upper = minimum(ys_true)
	y_max_upper = maximum(ys_true)
	y_margin = 0.05 * (y_max_upper - y_min_upper)
	y_min_upper -= y_margin
	y_max_upper += y_margin

	if separate_figures
		fig_func = Figure(size=(800, 500))
		fig_post = Figure(size=(800, 500))
		ax_func = Axis(fig_func[1, 1], title="Simulator", xlabel="Parameters x", ylabel="y = f(x)",
			titlesize=32, xlabelsize=28, ylabelsize=28, xticklabelsize=22, yticklabelsize=22)
		ax_post = Axis(fig_post[1, 1], title="Posterior Distribution", xlabel="Parameters x", ylabel="p(x|obs)",
			titlesize=32, xlabelsize=28, ylabelsize=28, xticklabelsize=22, yticklabelsize=22)
	else
		fig = Figure(size=(800, 1000))
		ax_func = Axis(fig[1, 1], title="Simulator", xlabel="Parameters x", ylabel="y = f(x)",
			titlesize=32, xlabelsize=28, ylabelsize=28, xticklabelsize=22, yticklabelsize=22)
		ax_post = Axis(fig[2, 1], title="Posterior Distribution", xlabel="Parameters x", ylabel="p(x|obs)",
			titlesize=32, xlabelsize=28, ylabelsize=28, xticklabelsize=22, yticklabelsize=22)
	end

	band!(ax_func, xs, fill(obs_lower, length(xs)), fill(obs_upper, length(xs)), color=(:orange, 0.2))
	hlines!(ax_func, [obs_mean], color=:orange, linewidth=2, label="observation ± 2σ")
	lines!(ax_func, xs, ys_true, color=:black, linewidth=3, linestyle=:dash, label="true function")
	if show_surrogate_in_simulator
		band!(ax_func, xs, ys_surrogate_lower, ys_surrogate_upper, color=(:blue, 0.2))
		lines!(ax_func, xs, ys_surrogate, color=:blue, linewidth=3, label="surrogate mean")
	end
	scatter!(ax_func, X_data, Y_data, color=:red, markersize=15, label="evaluations")
	ylims!(ax_func, y_min_upper, y_max_upper)
	axislegend(ax_func, position=:lt, labelsize=24)

	lines!(ax_post, xs, post_true, color=:black, linewidth=3, linestyle=:dash, label="true posterior")
	if !isempty(ci_intervals)
		y_ci = -0.03 * maximum(post_true)
		cap_height = 0.03 * maximum(post_true)
		for (i, (ci_lower, ci_upper)) in enumerate(ci_intervals)
			label = i == 1 ? "credible interval" : nothing
			lines!(ax_post, [ci_lower, ci_upper], [y_ci, y_ci], color=:green, linewidth=6, label=label)
			lines!(ax_post, [ci_lower, ci_lower], [y_ci - cap_height, y_ci + cap_height], color=:green, linewidth=6)
			lines!(ax_post, [ci_upper, ci_upper], [y_ci - cap_height, y_ci + cap_height], color=:green, linewidth=6)
		end
	end
	band!(ax_post, xs, post_approx_lower, post_approx_upper, color=(:blue, 0.2))
	lines!(ax_post, xs, post_approx, color=:blue, linewidth=3, label="approx. posterior")
	post_at_data = [approx_post([x])[1] for x in X_data]
	scatter!(ax_post, X_data, post_at_data, color=:red, markersize=15, label="evaluations")
	post_min = minimum(post_true)
	post_max = maximum(post_true)
	post_margin = 0.05 * (post_max - post_min)
	ylims!(ax_post, post_min - post_margin, post_max + post_margin)
	axislegend(ax_post, position=:lt, labelsize=24)

	if separate_figures
		return (; fig_func, fig_post)
	end

	return fig
end

@kwdef mutable struct BosipPlotCallback <: BosipCallback
	iter::Int = 0
    display_plots::Bool = true
	plot_dir::String = "./plots_bosip"
	save_plots::Bool = true
	separate_figures::Bool = false
	show_surrogate_in_simulator::Bool = true
	credible_interval_level::Union{Nothing, Float64} = nothing
end

function (cb::BosipPlotCallback)(problem::BosipProblem; first::Bool, options::BOSS.BossOptions, kwargs...)
	cb.iter += 1
	
	if cb.save_plots && first
        isdir(cb.plot_dir) && rm(cb.plot_dir; recursive=true)
		mkpath(cb.plot_dir)
	end
	
	fig = plot_state(problem; separate_figures=cb.separate_figures, show_surrogate_in_simulator=cb.show_surrogate_in_simulator, credible_interval_level=cb.credible_interval_level)
	
    if cb.display_plots
		if cb.separate_figures
			display(fig.fig_func)
			display(fig.fig_post)
		else
        	display(fig)
		end
    end

	if cb.save_plots
		if cb.separate_figures
			fig_func_path = joinpath(cb.plot_dir, "iter_$(cb.iter)_simulator.png")
			fig_post_path = joinpath(cb.plot_dir, "iter_$(cb.iter)_posterior.png")
			save(fig_func_path, fig.fig_func)
			save(fig_post_path, fig.fig_post)
			options.info && @info "Saved plots to $fig_func_path and $fig_post_path"
		else
			fig_path = joinpath(cb.plot_dir, "iter_$(cb.iter).png")
			save(fig_path, fig)
			options.info && @info "Saved plot to $fig_path"
		end
	end
end
