module SMC

export HiddenMarkovModel
export hmm_simulate
export hmm_log_marginal_likelihood
export hmm_posterior_sample
export hmm_log_joint_probability
export HMMPriorSMCScheme
export HMMConditionalSMCScheme
export logsumexp
export SMCScheme
export smc
export conditional_smc

include("mathutils.jl")
include("generic_smc.jl")
include("hmm.jl")
include("hmm_render.jl")

end
