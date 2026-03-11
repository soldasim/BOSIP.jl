### Define the problem

# Simple Linear — simulator returns (y, ∇y) via forward-mode adjoint.
# The GP does not know the gradient is constant (∇y ≡ 1); gradient noise
# is set large enough that it must learn this from data.

function f(x)
    y_func(x) = x                       # y : Rˣ → Rˣ, linear
    y = y_func(x)
    J = ForwardDiff.jacobian(y_func, x) # y_dim × x_dim Jacobian
    ∇y = vec(J')                        # stacked row-wise: [∂y₁/∂x₁,...,∂y₁/∂x_d, ...]
    return y, ∇y
end

bounds = ([-2.], [2.])
est_max_amplitude = 2.
z_obs = [0.]
std_obs = [0.2]
grad_noise_std = 0.3   # noisy gradient observations; GP must learn gradient magnitude from data


# Multimodal

# function f(x)
#     y_func(x) = [(x[1]^3 + 2*x[1]^2) / (abs(x[1])^3 + 3)]
#     y = y_func(x)
#     J = ForwardDiff.jacobian(y_func, x)
#     ∇y = vec(J')
#     return y, ∇y
# end

# bounds = ([-2.], [2.])
# est_max_amplitude = 2.
# z_obs = [0.3]
# std_obs = [0.05]
# grad_noise_std = 0.3
