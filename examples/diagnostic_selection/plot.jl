module Plot

using Plots
using Printf
using BOSS, BOLFI
using Distributions
include("toy_problem.jl")


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
    if plt.iters % plt.plot_each == 0
        options.info && @info "Plotting ..."
        title = "Iteration $(plt.iters)"
        plot_state(problem; term_cond=plt.term_cond, iter=plt.iters, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name="p_$(plt.iters)", put_in_scale=plt.put_in_scale, acquisition=acquisition.acq, title, separate=true)
    end
    plt.iters += 1
end


# - - - Plotting Scripts - - - - -

# size for basic text
const text_size = 16
# size for main headings
const heading_size = 20
# legend text
const legend_text_size = 14

# data scatter
const data_marker = (
    markersize = 5,
    markershape = :none,
    color = RGB(0., 0.5, 0.5),  # dark cyan
    markerstrokewidth = 0.4,
)
const new_point_marker = (
    markersize = 5,
    markershape = :none,
    color = RGB(0.5, 1.0, 1.0),  # light cyan
    markerstrokewidth = 0.4,
)
# post contours
const contour_kwargs = (
    color = cgrad(:matter; rev=true),
    level = 100,
)
# confidence regions
const conf_reg_gradient = [
    RGB(0.0, 0.5, 0.0),  # dark green
    RGB(1.0, 1.0, 0.0),  # yellow
    RGB(0.5, 1.0, 0.5),  # light green
]
const conf_reg_real = RGB(1.0, 0.0, 1.0)  # magenta
const conf_reg_expect = RGB(0.0, 1.0, 1.0)  # cyan
const conf_reg_approx = conf_reg_gradient[2]  # yellow
const conf_reg_kwargs = (
    linewidth = 2,
)
# feature rules
const rule_kwargs = (
    color = :white,
    linestyle = :dash,
    linewidth = 2,
    linealpha = 0.4,
)

# main titles
const bigtitle_kwargs = (titlefontsize=heading_size,) #titlefontfamily="New Century Schoolbook Bold"
const set_titles = [
    "FEATURE 1", #(θ₁ * θ₂ = 1)
    "FEATURE 2", #(θ₁ - θ₂ = 0)
]

# axis label annotation offsets
# (`+` is always outwards)
LO(::Val{:T1}) = [
    -0.3 +0.5  # xlabel
    +0.6 -0.2  # ylabel
]
LO(::Val{:T2}) = [
    -0.3 +0.5  # xlabel
    +0.7 -0.2  # ylabel
]

# write the iou value
annotate_iou!(p, t) = annotate!(p, 4.5, -4.5, (t, text_size, :white, :right))

function separate_new_datum(problem)
    bolfi = deepcopy(problem)
    new = bolfi.problem.data.X[:,end]
    bolfi.problem.data.X = bolfi.problem.data.X[:, 1:end-1]
    bolfi.problem.data.Y = bolfi.problem.data.Y[:, 1:end-1]
    return bolfi, new
end

function plot_state(problem; term_cond, iter, display=true, save_plots=false, plot_dir=".", plot_name="p", put_in_scale=false, acquisition, title=nothing, separate=false)
    if separate
        bolfi, new_datum = separate_new_datum(problem)
    else
        bolfi, new_datum = problem, nothing
    end
    step = 0.05 # TODO 0.05
    
    if problem isa BolfiProblem{Nothing}
        p = plot_sbi(bolfi; new_datum, iter, term_cond, put_in_scale, acquisition, step)
    else
        p = plot_sbfs(bolfi; new_datum, iter, term_cond, put_in_scale, acquisition, step)
    end
    
    plot!(p; dpi=100)  # (the plot is huge)
    display && Plots.display(p)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".pdf")
end

function plot_sbi(bolfi; new_datum=nothing, iter=nothing, term_cond, put_in_scale=false, acquisition, step=0.05)
    @assert acquisition isa PostVariance
    mode = :T1
    noise_vars_true = ToyProblem.σe_true(Val(mode)) .^ 2
    y_dim = ToyProblem.y_dim(Val(mode))

    title = "ITERATION $iter"
    prefixes = ["(a$iter) ", "(b$iter) ", "(c$iter) ", "(d$iter) "]

    p_ = plot_subset(bolfi; mode, title_prefixes=prefixes, new_datum, term_cond, display=false, put_in_scale, noise_vars_true, acquisition=PostVariance(), step, y_set=fill(true, y_dim), title)
    p = plot(p_; size=(540, 2322))
    return p
end

function plot_sbfs(bolfi; new_datum=nothing, iter=nothing, term_cond, put_in_scale=false, acquisition, step)
    @assert acquisition isa SetsPostVariance
    mode = :T2
    noise_vars_true = ToyProblem.σe_true(Val(mode)) .^ 2

    prefixes = [
        ["(a) ", "(c) ", "(e) ", "(g) "],
        ["(b) ", "(d) ", "(f) ", "(h) "],
    ]
    acq_prefix = "(i) "

    subset_plots = [plot_subset(get_subset(bolfi, bolfi.y_sets[:,i]); mode, title_prefixes=prefixes[i], new_datum, term_cond, display=false, put_in_scale, noise_vars_true=noise_vars_true[bolfi.y_sets[:,i]], acquisition=PostVariance(), step, y_set=bolfi.y_sets[:,i], title=set_titles[i]) for i in 1:size(bolfi.y_sets)[2]]
    acq_plot = plot_acquisition(bolfi; mode, title_prefix=acq_prefix, new_datum, acquisition, step)
    
    p_subsets = plot(subset_plots...; layout=grid(1, length(subset_plots)))
    s = plot(; framestyle=:none, bottom_margin=-100Plots.px)
    p = plot(p_subsets, s, acq_plot; layout=grid(3, 1; heights=[0.79, 0.02, 0.19]), size=(1080, 2916))
    return p
end

function plot_acquisition(bolfi; mode, title_prefix="", new_datum, acquisition, step=0.05)
    problem = bolfi.problem
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    acq = acquisition(bolfi, BolfiOptions())

    p4 = plot(; title=title_prefix*"MWMV Acquisition", colorbar=false, bigtitle_kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); mode, lims, label=nothing, step, acq_mode=true)
    plot_samples!(p4, X; mode, new_datum, label=nothing)

    LO_ = LO(Val(mode))
    annotate!(p4, +5 + LO_[1,1], -5 - LO_[1,2], text("θ₁", :left, text_size; color=:black))
    annotate!(p4, -5 - LO_[2,1], +5 + LO_[2,2], text("θ₂", :left, text_size; color=:black))
    
    s = plot(; framestyle=:none, bottom_margin=-100Plots.px)
    return plot(s, p4, s; layout=grid(1, 3; widths=[0.23, 0.54, 0.23]))
end

function plot_subset(bolfi; mode, title_prefixes=fill("", 4), new_datum=nothing, term_cond, display=true, put_in_scale=false, noise_vars_true, acquisition, step=0.05, y_set=fill(true, ToyProblem.y_dim), title=nothing)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood `p(y | x) * p(x) ∝ p(x | y)`
    function ll_post(x)
        y = ToyProblem.experiment(Val(mode), x; noise_vars=zeros(ToyProblem.y_dim(Val(mode))))[y_set]
        
        # ps = numerical_issues(x) ? 0. : 1.
        isnothing(y) && return 0.

        ll = pdf(MvNormal(y, sqrt.(noise_vars_true)), ToyProblem.y_obs(Val(mode))[y_set])
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # confidence sets
    if hasproperty(term_cond, :xs)
        xs = term_cond.xs
        @info "Loaded $(size(xs)[2]) samples."
    else
        samples = 10_000
        @info "Sampling $samples samples."
        xs = rand(x_prior, samples)
    end

    n = hasproperty(term_cond, :n) ? term_cond.n : 1.
    q = hasproperty(term_cond, :q) ? term_cond.q : 0.8
    if all(hasproperty.(Ref(term_cond), [:n, :q]))
        @info "Plotting with `q = $q` and `n = $n`."
    else
        q_sym = hasproperty(term_cond, :q) ? "=" : "←"
        n_sym = hasproperty(term_cond, :n) ? "=" : "←"
        @warn "Plotting with `q $q_sym $q` and `n $n_sym $n`.\nSome params are hard-coded! (marked by `←`)"
    end

    # helper func
    function post_and_cutoff(gp_post, x_prior, var_e, q; xs)
        post = posterior_mean(gp_post, x_prior, var_e; xs, normalize=true)
        post, c = find_cutoff(post, x_prior, q; xs)
        return post, c
    end

    post_real, c_real = find_cutoff(ll_post, x_prior, q; xs)  # unnormalized
    post_expect, c_expect = post_and_cutoff(gp_post, x_prior, bolfi.var_e, q; xs)
    post_approx, c_approx = post_and_cutoff(gp_bound(gp_post, 0.), x_prior, bolfi.var_e, q; xs)
    post_lb, c_lb = post_and_cutoff(gp_bound(gp_post, -n), x_prior, bolfi.var_e, q; xs)
    post_ub, c_ub = post_and_cutoff(gp_bound(gp_post, +n), x_prior, bolfi.var_e, q; xs)

    conf_sets_real = [
        (p=post_real, c=c_real, label="true posterior", color=conf_reg_real),
        (p=post_expect, c=c_expect, label="expect. posterior", color=conf_reg_expect),
        (p=post_approx, c=c_approx, label="approx. posterior", color=conf_reg_approx),
    ]
    conf_sets_approx = [
        (p=post_lb, c=c_lb, label="GP-LB posterior", color=conf_reg_gradient[1], linestyle=:dash),
        (p=post_approx, c=c_approx, label="approx. posterior", color=conf_reg_gradient[2]),
        (p=post_ub, c=c_ub, label="GP-UB posterior", color=conf_reg_gradient[3], linestyle=:dash),
    ]

    # calculate IoU ratios
    in_approx = (post_approx.(eachcol(xs)) .> c_approx)
    in_expect = (post_expect.(eachcol(xs)) .> c_expect)
    in_lb = (post_lb.(eachcol(xs)) .> c_lb)
    in_ub = (post_ub.(eachcol(xs)) .> c_ub)
    ae_ratio = set_iou(in_approx, in_expect, x_prior, xs)
    ublb_ratio = set_iou(in_ub, in_lb, x_prior, xs)
    
    # gp-approximated posterior likelihood
    ll_gp(a, b) = post_approx([a, b])

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())

    # - - - PLOT - - - - -
    clims = nothing
    if put_in_scale
        _, max_ll = maximize_prima(ll_post, x_prior, bounds; multistart=32, rhoend=1e-4)
        clims = (0., 1.2*max_ll)
    end
    kwargs = (colorbar=false,)

    p1 = plot(; title=title_prefixes[1]*"True Posterior", clims, kwargs...)
    plot_posterior!(p1, (a,b)->ll_post([a, b]); mode, conf_sets=conf_sets_real, lims, label=nothing, step)
    plot_samples!(p1, X; mode, new_datum, label=nothing)
    annotate_iou!(p1, "A-E IoU: $(@sprintf("%.4f", ae_ratio))")

    p2 = plot(; title=title_prefixes[2]*"Approx. Posterior", clims, kwargs...)
    plot_posterior!(p2, ll_gp; mode, conf_sets=conf_sets_approx, lims, label=nothing, step)
    plot_samples!(p2, X; mode, new_datum, label=nothing)
    annotate_iou!(p2, "UB-LB IoU: $(@sprintf("%.4f", ublb_ratio))")

    p3 = plot(; title=title_prefixes[3]*"Abs. Value of GP Mean", kwargs...)
    plot_posterior!(p3, (a,b) -> abs(gp_post([a,b])[1][1]); mode, lims, label=nothing, step)
    plot_samples!(p3, X; mode, new_datum, label=nothing)

    p4 = plot(; title=title_prefixes[4]*"Approx. Posterior Variance", kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); mode, lims, label=nothing, step)
    plot_samples!(p4, X; mode, new_datum, label=nothing)

    isnothing(title) && (title = put_in_scale ? "(in scale)" : "(not in scale)")
    t = plot(; title, framestyle=:none, bottom_margin=-160Plots.px, bigtitle_kwargs...)
    p = plot(t, p1, p2, p3, p4; layout=grid(1 + 4, 1; heights=[0.05, fill(0.95/4, 4)...]))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; mode, conf_sets=[], lims, label=nothing, step=0.05, acq_mode=false)
    xlabel = ""
    ylabel = ""
    ticks = -4:2:4

    # margins & axis ticks
    ytick_offset(::Val{:T2}) = 8
    ytick_offset(::Val{:T1}) = 5

    other = (
        left_margin = (mode == :T1) ? 0Plots.px : -5Plots.px,
        right_margin = 5Plots.px,
        top_margin = -10Plots.px,
        bottom_margin = -20Plots.px,

        aspect_ratio = :equal,
        widen = 1.02,

        xtickfontvalign = :bottom,
        ytickfonthalign = :left,
        yticks = (ticks, " "^ytick_offset(Val(mode)) .* string.(ticks)),
        xticks = (ticks, string.(ticks)),

        titlefontsize = text_size,
        legendfontsize = legend_text_size,
        tickfontsize = legend_text_size,

        legendfontcolor = :white,
        legend_foreground_color = nothing,
        legend_background_color = RGBA(0., 0., 0., 0.4),
    )

    grid = lims[1]:step:lims[2]
    vals = ll.(grid', grid)
    clims = (minimum(vals), maximum(vals))
    contourf!(p, grid, grid, vals; clims, xlabel, ylabel, contour_kwargs..., other...)

    if !acq_mode
        # axis labels
        LO_ = LO(Val(mode))
        annotate!(p, +5 + LO_[1,1], -5 - LO_[1,2], text("θ₁", :left, text_size; color=:black))
        annotate!(p, -5 - LO_[2,1], +5 + LO_[2,2], text("θ₂", :left, text_size; color=:black))
    end

    # "OBSERVATION-RULES"
    plot!(p, a->1/a, grid; y_lims=lims, label, rule_kwargs...)
    plot!(p, a->a, grid; y_lims=lims, label, rule_kwargs...)

    # CONFIDENCE SET
    target = mean(vals)
    for (f, c, kwargs...) in conf_sets
        plot_confidence_set!(p, f, c; mode, target, lims, step, conf_reg_kwargs..., kwargs...)
    end

    plot!(p; tickfonstize = legend_text_size)
    return p
end

function plot_samples!(p, samples; mode, new_datum=nothing, label=nothing)
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, data_marker...)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, new_point_marker...)
end

function plot_confidence_set!(p, ll, c; mode, target, lims, step, label=nothing, color=:red, kwargs...)
    grid = lims[1]:step:lims[2]
    norm = target / c  # s.t. the contour will be within climss
    contour!(p, grid, grid, (a,b)->norm*ll([a,b]); levels=[norm*c], color, kwargs...)
    isnothing(label) || scatter!(p, [], []; label, color, markerstrokewidth=0.4, markerstrokealpha=0.)
end

end # module Plot
