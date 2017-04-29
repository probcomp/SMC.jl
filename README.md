# Implementation of sequential Monte Carlo in Julia

Includes generic SMC and conditional SMC algorithms, and
specific instantiations of algorithms for HMMs.

Also includes some infrastructure and tests for divergence measurement

Tested with Julia Version 0.5.1 (2017-03-05 13:25 UTC)

# Dependencies

```
julia> Pkg.add("Distributions")
```

# To run tests:

```
julia
julia> include("test/runtests.jl")

```
