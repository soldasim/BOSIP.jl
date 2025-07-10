
function plot_param_slices(plt::PlotCallback, bolfi::BolfiProblem; samples=20_000, options=BolfiOptions(), kwargs...)
    return plot_param_slices(bolfi; samples, options, plt.display, plt.step)
end

function plot_param_slices(bolfi::BolfiProblem; options=BolfiOptions(), samples=20_000, display=true, step=0.05)
    options.info && @info "Plotting ..."
    x_prior = bolfi.x_prior
    
    @assert x_prior isa Product  # individual priors must be independent
    param_samples = rand(x_prior, samples)

    return [plot_param_post(bolfi, i, param_samples; display, step) for i in 1:BOLFI.x_dim(bolfi)]
end

function plot_param_post(bolfi::BolfiProblem, param_idx::Int, param_samples::AbstractMatrix{<:Real}; display=true, step=0.05)
    bounds = bolfi.problem.domain.bounds
    x_prior = bolfi.x_prior
    param_range = bounds[1][param_idx]:step:bounds[2][param_idx]
    param_samples_ = deepcopy(param_samples)
    title = "Param $param_idx posterior"

    # true posterior
    function true_post(x)
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))
        ll = pdf(MvNormal(y, ToyProblem.σe_true), ToyProblem.z_obs)
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
    f = Figure()
    ax = Axis(f[1,1])
    lines!(ax, param_range, true_post_slice; label="true posterior")
    lines!(ax, param_range, exp_post_slice; label="expected posterior")
    display && Base.display(f)
    return f
end
