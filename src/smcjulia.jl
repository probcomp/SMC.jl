module SMCJulia

    using Distributions
    using Base.Test
    
    include("mathutils.jl")
    include("hmm.jl")
    include("smc.jl")
    include("tests.jl")

end
