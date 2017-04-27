immutable HiddenMarkovModel
    # NOTE: all must be appropriately normalized
    initial_state_prior::Array{Float64,1}
    transition_model::Array{Float64,2}
    observation_model::Array{Float64,2}
end

function simulate(hmm::HiddenMarkovModel, num_steps::Int)
    states = Array{Int, 1}(num_steps)
    observations = Array{Int, 1}(num_steps)
    states[1] = rand(Categorical(hmm.initial_state_prior))
    for i=2:num_steps
        prev = states[i-1]
        states[i] = rand(Categorical(hmm.transition_model[prev,:]))
    end
    for i=1:num_steps
        observations[i] = rand(Categorical(hmm.observation_model[states[i],:]))
    end
    (states, observations)
end

function marginal_likelihood(hmm::HiddenMarkovModel, observations::Array{Int, 1})
    # TODO
end

function posterior_sample(hmm::HiddenMarkovModel, observations::Array{Int, 1})
    # TODO [[ combine with marginal likelihood ? ] -- they probably hvae common comp.
end


immutable HMMPriorInitializer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(init::HMMPriorInitializer)
    rand(Categorical(init.hmm.initial_state_prior))
end

function weight(init::HMMPriorInitializer, x::Int)
    # the likelihood of the observation given state x
    pdf(Categorical(init.hmm.observation_model[x,:]), observation)
end

immutable HMMPriorIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(incr::HMMPriorIncrementer, x::Int)
    rand(Categorical(incr.hmm.transition_model[x,:]))
end

function weight(init::HMMPriorIncrementer, x::Int)
    pdf(Categorical(init.hmm.observation_model[x,:]), observation)
end

