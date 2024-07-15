module Plot

using Plots
using Distributions
using BOSS, BOLFI

include("toy_problem.jl")


# - - - Plotting Callback - - - - -

mutable struct PlotCallback <: BolfiCallback
    prev_state::Union{Nothing, BolfiProblem}
    iters::Int
    plot_each::Int
    display::Bool
    save_plots::Bool
    plot_dir::String
    plot_name::String
    noise_std_true::Vector{Float64}
end
PlotCallback(;
    plot_each::Int = 1,
    display::Bool = true,
    save_plots::Bool = false,
    plot_dir::String = ".",
    plot_name::String = "p",
    noise_std_true::Vector{Float64},
) = PlotCallback(nothing, 0, plot_each, display, save_plots, plot_dir, plot_name, noise_std_true)

"""
Plots the state in the current iteration.
"""
function (plt::PlotCallback)(bolfi::BolfiProblem; acquisition, options, first, kwargs...)
    if first
        plt.prev_state = deepcopy(bolfi)
        plt.iters += 1
        return
    end
    
    # `iters - 1` because the plot is "one iter behind"
    plot_iter = plt.iters - 1

    if plot_iter % plt.plot_each == 0
        options.info && @info "Plotting ..."
        new_datum = bolfi.problem.data.X[:,end]
        plot_state(plt.prev_state, new_datum; display=plt.display, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", noise_std_true=plt.noise_std_true, acquisition=acquisition.acq)
    end
    
    plt.prev_state = deepcopy(bolfi)
    plt.iters += 1
end

"""
Plot the final state after the BO procedure concludes.
"""
function plot_final(plt::PlotCallback; acquisition, options, kwargs...)
    plot_iter = plt.iters - 1
    options.info && @info "Plotting ..."
    plot_state(plt.prev_state, nothing; display=plt.display, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", noise_std_true=plt.noise_std_true, acquisition)
end

"""
Plot real and approximate posteriors of each individual parameter.
"""
function plot_param_slices(plt::PlotCallback; samples=20_000, display=true, step=0.05)
    # implented for independent priors only (for now)
    bolfi = plt.prev_state
    x_prior = bolfi.x_prior
    
    @assert x_prior isa Product  # individual priors must be independent
    param_samples = rand(x_prior, samples)

    return [plot_param_post(bolfi, i, param_samples; display, step) for i in 1:BOLFI.x_dim(bolfi)]
end


# - - - Initialization - - - - -

init_plotting(plt::PlotCallback) =
    init_plotting(; save_plots=plt.save_plots, plot_dir=plt.plot_dir)

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            rm(plot_dir, recursive=true)
        end
        mkdir(plot_dir)
    end
end


# - - - Plot State - - - - -

function plot_state(bolfi, new_datum; display=true, save_plots=false, plot_dir=".", plot_name="p", noise_std_true, acquisition)
    # Plots with hyperparams fitted using *all* data! (Does not really matter.)
    p = plot_samples(bolfi; new_datum, display, noise_std_true, acquisition)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_samples(bolfi; new_datum=nothing, display=true, noise_std_true, acquisition)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    step = 0.05
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood `p(d | a, b) * p(a, b) ∝ p(a, b | d)`
    function ll_post(a, b)
        x = [a, b]
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))

        ll = pdf(MvNormal(y, noise_std_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # gp-approximated posterior likelihood
    post_μ = BOLFI.posterior_mean(gp_post, x_prior, bolfi.std_obs; normalize=false)
    ll_gp(a, b) = post_μ([a, b])

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())

    # - - - PLOT - - - - -
    kwargs = (colorbar=false,)

    p1 = plot(; title="true posterior", kwargs...)
    plot_posterior!(p1, ll_post; ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p1, X; new_datum, label=nothing)

    p2 = plot(; title="posterior mean", kwargs...)
    plot_posterior!(p2, ll_gp; ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p2, X; new_datum, label=nothing)

    p3 = plot(; title="abs. value of GP mean", kwargs...)
    plot_posterior!(p3, (a,b) -> abs(gp_post([a,b])[1][1]); ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p3, X; new_datum, label=nothing)

    p4 = plot(; title="posterior variance", kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p4, X; new_datum, label=nothing)

    title = ""
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; y_obs, lims, label="ab=d", step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, ll)
    
    # "OBSERVATION-RULES"
    obs_color = :gold
    plot!(p, a->y_obs[1]/a, grid; y_lims=lims, label, color=obs_color)
end

function plot_samples!(p, samples; new_datum=nothing, label="(a,b) ~ p(a,b|d)")
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
end


# - - - Plot Parameter Slices - - - - -

function plot_param_post(bolfi, param_idx, param_samples; display, step=0.05)
    bounds = bolfi.problem.domain.bounds
    x_prior = bolfi.x_prior
    param_range = bounds[1][param_idx]:step:bounds[2][param_idx]
    param_samples_ = deepcopy(param_samples)
    title = "Param $param_idx posterior"

    # true posterior
    function true_post(x)
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))
        ll = pdf(MvNormal(y, ToyProblem.σe_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end
    py = evidence(true_post, x_prior; xs=param_samples)
    function true_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return mean(true_post.(eachcol(param_samples_))) / py
    end

    # expected posterior
    exp_post = BOLFI.posterior_mean(bolfi; normalize=true)
    function exp_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return exp_post.(eachcol(param_samples_)) |> mean
    end

    # plot
    p = plot(; title)
    plot!(p, true_post_slice, param_range; label="true posterior")
    plot!(p, exp_post_slice, param_range; label="expected posterior")
    display && Plots.display(p)
    return p
end


end # module Plot
