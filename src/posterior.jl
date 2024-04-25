
function posterior_mean(x_prior, gp_post; var_e, py=1.)
    function mean(x)
        pred = gp_post(x)
        μ_gp, var_gp = pred[1], pred[2]
        y_dim = length(μ_gp)
        ll = pdf(MvNormal(μ_gp, sqrt.(var_e .+ var_gp)), zeros(y_dim))
        pθ = pdf(x_prior, x)
        return (pθ / py) * ll
    end
end

function posterior_variance(x_prior, gp_post; var_e, py=1.)
    function var(x)
        pred = gp_post(x)
        μ_y, var_y = pred[1], pred[2]
        
        # observation noise: var_e ≈ σe^2  # approx. as scalar hyperparameter
        # simulation noise: var_gp ≈ ω^2   # approx. by GP noise

        prodA = log.(A_.(var_e, μ_y, var_y)) |> sum |> exp
        prodB = log.(B_.(var_e, μ_y, var_y)) |> sum |> exp
        pθ = pdf(x_prior, x)
        return (pθ^2 / py^2) * (prodA - prodB)
    end
end

function A_(var_e, μ_y, var_y)
    varA = var_e + 2*var_y
    return sqrt(varA / var_e) * pdf(Normal(0, sqrt(varA)), μ_y)^2
end

function B_(var_e, μ_y, var_y)
    varB = var_e + var_y
    return pdf(Normal(0, sqrt(varB)), μ_y)^2
end
