# using Pkg
# Pkg.activate("examples")

using BOSS, BOSIP
using Distributions
using OptimizationPRIMA
using GLMakie
using FileIO

include("toy_problem.jl")

function true_post(x)
    y = ToyProblem.simulation(x; noise_std=zeros(ToyProblem.y_dim))
    log_py = logpdf(MvNormal(y, ToyProblem.σe), ToyProblem.z_obs)
    log_px = logpdf(ToyProblem.get_x_prior(), x)
    return exp(log_py + log_px)
end

function init_problem()
    X = hcat(mean(ToyProblem.get_bounds()))
    Y = hcat(ToyProblem.simulation.(eachcol(X))...)
    
    bosip = ToyProblem.bosip_problem(ExperimentData(X, Y))
    BOSIP._init_problem!(bosip, BosipOptions())
    
    return bosip
end

function plot_confidence_contour!(ax, xs; q=0.9)
    bounds = ToyProblem.get_bounds()
    x = range(bounds[1][1], bounds[2][1], length=200)
    y = range(bounds[1][2], bounds[2][2], length=200)
    z = [true_post([xi, yi]) for xi in x, yi in y]

    log_prior = logpdf.(Ref(ToyProblem.get_x_prior()), eachcol(xs))
    log_post = log.(true_post.(eachcol(xs)))

    ws = exp.( log_post .-  log_prior)
    ws[log_post .== -Inf] .= 0.0
    c = find_cutoff(true_post, xs, ws, q)

    contour!(ax, x, y, z, levels=[c], color=:red, linewidth=2)
    text!(ax, 4.8, 0., text="True Posterior", color=:red, align=(:right, :top), fontsize=16)
end

function plot_state!(fig, bosip::BosipProblem; hint=true, xs=nothing, uncertainty=true)
    bounds = bosip.problem.domain.bounds
    post_mean = posterior_mean(bosip)
    post_var = posterior_variance(bosip)
    X = bosip.problem.data.X

    if isnothing(xs)
        xs = rand(bosip.x_prior, 10_000)
    end

    x = range(bounds[1][1], bounds[2][1], length=200)
    y = range(bounds[1][2], bounds[2][2], length=200)

    # Axis 1: expected posterior
    if isempty(contents(fig[1, 1]))
        ax = Axis(fig[1, 1], aspect=DataAspect(), title="Expected Posterior: p(x|y=1)", xlabel="x1", ylabel="x2")
        deregister_interaction!(ax, :rectanglezoom)
    else
        ax = content(fig[1, 1])
        empty!(ax)
    end
    z_post = [post_mean([xi, yi]) for xi in x, yi in y]
    if extrema(z_post) == (z_post[1], z_post[1]) # Check if all values are the same
        z_post[1, 1] += 1e-6 # Add a small perturbation to avoid single-value issue
    end
    hm = contourf!(ax, x, y, z_post, colormap=:viridis)
    hint && plot_confidence_contour!(ax, xs)
    scatter!(ax, X[1,:], X[2,:], color=:white, markersize=10)
    scatter!(ax, X[1,:], X[2,:], color=:black, markersize=8)

    gp_post = BOSS.model_posterior(bosip.problem)
    gp_μ(x) = mean(gp_post, x)[1]
    gp_σ(x) = std(gp_post, x)[1]

    # Axis 2: GP prediction
    if isempty(contents(fig[1, 2]))
        ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Prediction for y=f(x)", xlabel="x1", ylabel="x2")
        deregister_interaction!(ax2, :rectanglezoom)
    else
        ax2 = content(fig[1, 2])
        empty!(ax2)
    end
    z_gp = [gp_μ([xi, yi]) for xi in x, yi in y]
    if extrema(z_gp) == (z_gp[1], z_gp[1]) # Check if all values are the same
        z_gp[1, 1] += 1e-6 # Add a small perturbation to avoid single-value issue
    end
    hm_gp = contourf!(ax2, x, y, z_gp, colormap=:vik, levels=-25:1:25)
    tightlimits!(ax2)
    scatter!(ax2, X[1,:], X[2,:], color=:white, markersize=10)
    scatter!(ax2, X[1,:], X[2,:], color=:black, markersize=8)

    if isempty(contents(fig[1, 3]))
        Colorbar(fig[1, 3], hm_gp)
    end

    # Contour where simulation(x) == 1
    if hint
        z_simulation = [ToyProblem.simulation([xi, yi])[1] for xi in x, yi in y]
        contour!(ax2, x, y, z_simulation, levels=[ToyProblem.z_obs[1]], color=:green, linestyle=:solid, linewidth=2)
        text!(ax2, 4.8, 0., text="y = 1", color=:green, align=(:right, :top), fontsize=16)
    end

    # Contour where f_gp equals ToyProblem.z_obs
    z_contour = [gp_μ([xi, yi]) for xi in x, yi in y]
    contour!(ax2, x, y, z_contour, levels=[ToyProblem.z_obs[1]], color=:yellow, linestyle=:dash, linewidth=2)

    if uncertainty
        # Axis 3: posterior variance
        if isempty(contents(fig[2, 1]))
            ax3 = Axis(fig[2, 1], aspect=DataAspect(), title="Posterior Uncertainty", xlabel="x1", ylabel="x2")
            deregister_interaction!(ax3, :rectanglezoom)
        else
            ax3 = content(fig[2, 1])
            empty!(ax3)
        end
        z_var = [post_var([xi, yi]) for xi in x, yi in y]
        if extrema(z_var) == (z_var[1], z_var[1]) # Check if all values are the same
            z_var[1, 1] += 1e-6 # Add a small perturbation to avoid single-value issue
        end
        hm_var = contourf!(ax3, x, y, z_var, colormap=:plasma)
        scatter!(ax3, X[1,:], X[2,:], color=:white, markersize=10)
        scatter!(ax3, X[1,:], X[2,:], color=:black, markersize=8)

        # Axis 4: GP standard deviation
        if isempty(contents(fig[2, 2]))
            ax4 = Axis(fig[2, 2], aspect=DataAspect(), title="Uncertainty of f(x)", xlabel="x1", ylabel="x2")
            deregister_interaction!(ax4, :rectanglezoom)
        else
            ax4 = content(fig[2, 2])
            empty!(ax4)
        end
        z_gp_std = [gp_σ([xi, yi]) for xi in x, yi in y]
        if extrema(z_gp_std) == (z_gp_std[1], z_gp_std[1]) # Check if all values are the same
            z_gp_std[1, 1] += 1e-6 # Add a small perturbation to avoid single-value issue
        end
        hm_gp_std = contourf!(ax4, x, y, z_gp_std, colormap=:plasma)
        scatter!(ax4, X[1,:], X[2,:], color=:white, markersize=10)
        scatter!(ax4, X[1,:], X[2,:], color=:black, markersize=8)
    end

    return fig
end

"""
    app(; kwargs...)

# Keywords

- `uncertainty::Bool=true`: Whether to show posterior uncertainty plots.
- `animate::Bool=false`: Whether to run the app in animation mode.
- `hint::Bool=false`: Whether to show the hint contour for the true posterior at the start.
"""
function app(; uncertainty=true, animate=false, hint=false)
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 20,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 20,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )

    # init
    p = init_problem()
    estimate_parameters!(p, model_fitter)

    xs = rand(p.x_prior, 10_000)

    fig = Figure(size=(1200, 800))
    plot_state!(fig, p; hint, xs, uncertainty)
    display(fig)

    if animate
        # ANIMATION MODE

        while true
            sleep(1.0)  # Wait for 1 second
            
            if length(p.problem.data.X[1, :]) >= 35
                p = init_problem()
                estimate_parameters!(p, model_fitter)
            else
                x = maximize_acquisition(p, acq_maximizer)
                eval_objective!(p, x)
                estimate_parameters!(p, model_fitter)
            end

            plot_state!(fig, p; hint, xs, uncertainty)
        end

    else
        # INTERACTIVE MODE

        button_bar = GridLayout(tellwidth=false)
        btn_next = Button(fig, label="Next Iteration")
        btn_reset = Button(fig, label="Reset")
        btn_hint = Button(fig, label="Toggle Hint")

        fig[2 + uncertainty, 1:2] = button_bar
        button_bar[1, 1] = btn_next
        button_bar[1, 2] = btn_reset
        button_bar[1, 3] = btn_hint
        
        # Button callbacks
        on(btn_next.clicks) do _
            x = maximize_acquisition(p, acq_maximizer)
            eval_objective!(p, x)
            estimate_parameters!(p, model_fitter)
            plot_state!(fig, p; hint, xs, uncertainty)
        end

        on(btn_reset.clicks) do _
            p = init_problem()
            estimate_parameters!(p, model_fitter)
            plot_state!(fig, p; hint, xs, uncertainty)
        end

        on(btn_hint.clicks) do _
            hint = !hint
            plot_state!(fig, p; hint, xs, uncertainty)
        end

        # Add interactivity to the plot
        ax1 = content(fig[1, 1])  # Retrieve the axis object
        register_interaction!(ax1, :add_point_1) do event::MouseEvent, axis
            if event.type === MouseEventTypes.leftclick
                origin = to_world(ax1.scene, ax1.scene.viewport.val.origin)
                widths = to_world(ax1.scene, ax1.scene.viewport.val.widths)
                click = to_world(ax1.scene, events(ax1).mouseposition.val)

                coords = [(click[1] - origin[1]) / widths[1], (click[2] - origin[2]) / widths[2]] ./ 2
                coords = coords .* (ToyProblem.get_bounds()[2] - ToyProblem.get_bounds()[1]) .+ ToyProblem.get_bounds()[1]
                
                println("Clicked coordinates: ", coords)

                if BOSS.in_bounds(coords, ToyProblem.get_bounds()) && !any(isapprox.(Ref(coords), eachcol(p.problem.data.X); atol=1e-2))
                    eval_objective!(p, coords)
                    estimate_parameters!(p, model_fitter)
                    plot_state!(fig, p; hint, xs, uncertainty)
                end
            end
        end

        # Add interactivity to the plot
        ax2 = content(fig[1, 2])  # Retrieve the axis object
        register_interaction!(ax2, :add_point_2) do event::MouseEvent, axis
            if event.type === MouseEventTypes.leftclick
                origin = to_world(ax2.scene, ax2.scene.viewport.val.origin)
                widths = to_world(ax2.scene, ax2.scene.viewport.val.widths)
                click = to_world(ax2.scene, events(ax2).mouseposition.val)

                coords = [(click[1] - origin[1]) / widths[1], (click[2] - origin[2]) / widths[2]] ./ 2
                coords = coords .* (ToyProblem.get_bounds()[2] - ToyProblem.get_bounds()[1]) .+ ToyProblem.get_bounds()[1]
                
                if BOSS.in_bounds(coords, ToyProblem.get_bounds()) && !any(isapprox.(Ref(coords), eachcol(p.problem.data.X); atol=1e-2))
                    eval_objective!(p, coords)
                    estimate_parameters!(p, model_fitter)
                    plot_state!(fig, p; hint, xs, uncertainty)
                end
            end
        end

        if uncertainty
            # Add interactivity to the uncertainty plots
            ax3 = content(fig[2, 1])  # Retrieve the axis object for posterior uncertainty
            register_interaction!(ax3, :add_point_3) do event::MouseEvent, axis
                if event.type === MouseEventTypes.leftclick
                    origin = to_world(ax3.scene, ax3.scene.viewport.val.origin)
                    widths = to_world(ax3.scene, ax3.scene.viewport.val.widths)
                    click = to_world(ax3.scene, events(ax3).mouseposition.val)

                    coords = [(click[1] - origin[1]) / widths[1], (click[2] - origin[2]) / widths[2]] ./ 2
                    coords = coords .* (ToyProblem.get_bounds()[2] - ToyProblem.get_bounds()[1]) .+ ToyProblem.get_bounds()[1]

                    if BOSS.in_bounds(coords, ToyProblem.get_bounds()) && !any(isapprox.(Ref(coords), eachcol(p.problem.data.X); atol=1e-2))
                        eval_objective!(p, coords)
                        estimate_parameters!(p, model_fitter)
                        plot_state!(fig, p; hint, xs, uncertainty)
                    end
                end
            end

            ax4 = content(fig[2, 2])  # Retrieve the axis object for GP standard deviation
            register_interaction!(ax4, :add_point_4) do event::MouseEvent, axis
                if event.type === MouseEventTypes.leftclick
                    origin = to_world(ax4.scene, ax4.scene.viewport.val.origin)
                    widths = to_world(ax4.scene, ax4.scene.viewport.val.widths)
                    click = to_world(ax4.scene, events(ax4).mouseposition.val)

                    coords = [(click[1] - origin[1]) / widths[1], (click[2] - origin[2]) / widths[2]] ./ 2
                    coords = coords .* (ToyProblem.get_bounds()[2] - ToyProblem.get_bounds()[1]) .+ ToyProblem.get_bounds()[1]

                    if BOSS.in_bounds(coords, ToyProblem.get_bounds()) && !any(isapprox.(Ref(coords), eachcol(p.problem.data.X); atol=1e-2))
                        eval_objective!(p, coords)
                        estimate_parameters!(p, model_fitter)
                        plot_state!(fig, p; hint, xs, uncertainty)
                    end
                end
            end
        end

    end
    
    return fig
end

function create_animation(; hint=true, uncertainty=true)
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 20,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 20,
        parallel = true,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )

    # Initialize the problem and parameters
    p = init_problem()
    estimate_parameters!(p, model_fitter)

    xs = rand(p.x_prior, 10_000)
    frames = []

    # Create a figure and plot the initial state
    fig = Figure(size=(1200, 800))
    plot_state!(fig, p; hint, xs, uncertainty)

    # Iterate and capture frames
    record(fig, "animation.mp4", 1:35; framerate=1) do _
        x = maximize_acquisition(p, acq_maximizer)
        eval_objective!(p, x)
        estimate_parameters!(p, model_fitter)
        plot_state!(fig, p; hint, xs, uncertainty)
    end
end

# app()
