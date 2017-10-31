# Implementation of sequential Monte Carlo in Julia

Includes generic SMC and conditional SMC algorithms, and
specific instantiations of algorithms for HMMs.

Also includes some infrastructure and tests for divergence measurement

Tested with Julia Version 0.6.0

# Dependencies

```
julia> Pkg.add("Distributions")
julia> Pkg.add("PyPlot")
```

# To run tests:

```
julia
julia> include("test/runtests.jl")

```
