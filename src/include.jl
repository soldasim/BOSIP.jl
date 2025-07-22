# Types
include("types/include.jl")

# Code
include("utils/include.jl")
include("posterior/include.jl")
include("sampling.jl")
include("main.jl")
include("plots.jl")
include("bi_acquisition.jl")
include("subset_problem.jl")

# Modules
include("likelihoods/include.jl")
include("acquisitions/include.jl")
include("term_conds/include.jl")
include("samplers/include.jl")
include("metrics/include.jl")
include("callbacks/include.jl")

# Other
include("deprecated.jl")
