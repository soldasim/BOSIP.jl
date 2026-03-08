### Define the problem

# Simple Linear

function f(x)
    return x
end

bounds = ([-2.], [2.])
est_max_amplitude = 2.
z_obs = [0.]
std_obs = [0.2]


# Multimodal

# function f(x)
#     x = x[1] 
#     y = (x^3 + 2*x^2) / (abs(x)^3 + 3)
#     return [y]
# end

# bounds = ([-2.], [2.])
# est_max_amplitude = 2.
# z_obs = [0.3]
# std_obs = [0.05]
