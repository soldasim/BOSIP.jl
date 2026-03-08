
Run the app as:
- `] activate /examples`
- `include("examples/interactive/app.jl")`
- `app(; kwargs...)`

Create an animation as:
- `] activate /examples`
- `include("examples/interactive/app.jl")`
- `create_animation()`

### Keywords
- `uncertainty::Bool`: Whether or not to plot the uncertainty plots.
- `animate::Bool`: Whether to run in animation mode or interactive mode.
- `hint::Bool`: Whether to start with the hint shown.
