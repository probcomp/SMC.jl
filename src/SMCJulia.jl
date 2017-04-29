module SMCJulia

    include("mathutils.jl")
    include("hmm.jl")
    include("smc.jl")
    
    # TODO: move to a generic math package?
    export logsumexp

    # Generic SMC stuff
    export SMCScheme
    export smc
    export conditional_smc

    # HMM stuff
    export HiddenMarkovModel
    export simulate
    export log_marginal_likelihood
    export posterior_sample
    export HMMPriorSMCScheme
    export HMMConditionalSMCScheme

end
