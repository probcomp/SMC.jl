# A Julia implementation of sequential Monte Carlo

Includes generic SMC and conditional SMC algorithms, and
specific instantiations of algorithms for HMMs.

WARNING: Gen is unsupported research software.

Tested with Julia Version 0.6.0

# Dependencies

```
julia> Pkg.add("Distributions")
julia> Pkg.add("PyPlot")
```

# Installation

SMC.jl is not a publicly registered Julia package.
To install:

```
julia> Pkg.clone("git@github.com:probcomp/SMC.jl.git")
```

# To run tests:

```
julia> Pkg.test("SMC")
```
