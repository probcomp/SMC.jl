module SMC

export HiddenMarkovModel
export num_states
export num_observations
export hmm_simulate
export hmm_log_marginal_likelihood
export hmm_posterior_sample
export hmm_log_joint_probability
export HMMPriorSMCScheme
export HMMConditionalSMCScheme
export render_hmm!
export render_hmm_states!
export render_hmm_posterior_marginals!
export render_hmm_observations!

export logsumexp
export SMCScheme
export smc
export conditional_smc

include("mathutils.jl")
include("generic_smc.jl")
include("hmm.jl")
include("hmm_render.jl")

end
