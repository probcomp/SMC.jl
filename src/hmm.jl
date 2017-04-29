immutable HiddenMarkovModel
    initial_state_prior::Array{Float64,1}
    transition_model::Array{Float64,2}
    observation_model::Array{Float64,2}
    num_states::Int
    num_obs::Int

    function HiddenMarkovModel(initial_state_prior, transition_model, observation_model)
        if sum(initial_state_prior) != 1.
            error("initial_state_prior is not normalized: $(initial_state_prior)")
        end
        num_states = size(initial_state_prior)[1]
        if size(transition_model) != (num_states, num_states)
            error("transition_model has wrong dimension: got $(size(transition_model)) expected $((num_states, num_states))")
        end
        if !isapprox(sum(transition_model, 2), ones(num_states, 1))
            error("transition_model rows are not all normalized: got $(sum(transition_model, 2))")
        end
        if size(observation_model)[1] != num_states
            error("observation_model has wrong number of rows: got $(size(observation_model)[1]) expected $num_states")
        end
        num_obs = size(observation_model)[2]
        if !isapprox(sum(observation_model, 2), ones(num_states, 1))
            error("observation_model rows are not all normalized: got $(sum(observation_model, 2))")
        end
        new(initial_state_prior, transition_model, observation_model, num_states, num_obs)
    end
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

function forward_pass(hmm::HiddenMarkovModel, observations::Array{Int, 1})
    # compute the forward probabilities
    # p(x_t, y_{1:t-1}) for t=1,...,T
    # NOTE: we are assuming an observation for each time point
    num_steps = length(observations)
    num_states = hmm.num_states
    fprobs = Array{Float64,2}(num_steps, num_states)
    fprobs[1,:] = hmm.initial_state_prior
    for t=2:num_steps
        obs = observations[t-1]
        message = fprobs[t-1,:] .* hmm.observation_model[:,obs]
        # rows are x_{t-1}, cols are x_t
        product = message .* hmm.transition_model
        fprobs[t,:] = vec(sum(product, 1)) # sum over rows
    end
    return fprobs
end

function log_forward_pass(hmm::HiddenMarkovModel, observations::Array{Int, 1})
    # compute the log forward probabilities
    # log p(x_t, y_{1:t-1}) for t=1,...,T
    # NOTE: we are assuming an observation for each time point
    num_steps = length(observations)
    num_states = hmm.num_states
    lfprobs = Array{Float64,2}(num_steps, num_states)
    lfprobs[1,:] = log(hmm.initial_state_prior)
    for t=2:num_steps
        obs = observations[t-1]
        lmessage = lfprobs[t-1,:] .+ log(hmm.observation_model[:,obs])
        # rows are x_{t-1}, cols are x_t; sum over rows
        lproduct = lmessage .+ log(hmm.transition_model)
        lfprobs[t,:] = logsumexp(lproduct, 1)
    end
    return lfprobs
end

function marginal_likelihood(hmm::HiddenMarkovModel, 
                             observations::Array{Int, 1})
    # use the forward probability p(x_T, y_{1:T-1})
    fprobs = forward_pass(hmm, observations)
    # p(y_{1:T}) = sum_{x_T} p(x_T, y_{1:T-1}) p(y_T | x_T)
    sum(fprobs[end,:] .* hmm.observation_model[:,observations[end]])
end

function log_marginal_likelihood(hmm::HiddenMarkovModel, 
                                 observations::Array{Int, 1})
    lfprobs = log_forward_pass(hmm, observations)
    logsumexp(lfprobs[end,:] .+ log(hmm.observation_model[:,observations[end]]))
end

function posterior_sample(hmm::HiddenMarkovModel, 
                          observations::Array{Int, 1})
    num_steps = length(observations)
    lfprobs = log_forward_pass(hmm, observations)
    ldist = lfprobs[end,:] .+ log(hmm.observation_model[:,observations[end]])
    particle = Array{Int,1}(num_steps)
    particle[end] = rand(Categorical(exp(ldist - logsumexp(ldist)))) # TODO there may be a more numerically precise version
    for t = num_steps-1:-1:1
        ldist = lfprobs[t,:]
        ldist = ldist .+ log(hmm.observation_model[:,observations[t]])
        ldist = ldist .+ log(hmm.transition_model[:,particle[t+1]])
        dist = exp(ldist - logsumexp(ldist))
        particle[t] = rand(Categorical(dist))
    end
    particle 
end

# -- Helper functions for SMC in HMM using prior proposals

immutable HMMPriorInitializer
    hmm::HiddenMarkovModel
    observation::Int # should be the first observation
end

function sample(init::HMMPriorInitializer)
    rand(Categorical(init.hmm.initial_state_prior))
end

function log_weight(init::HMMPriorInitializer, cur::Int)
    # the likelihood of the observation given state cur was output from sample
    log(init.hmm.observation_model[cur, init.observation])
end

immutable HMMPriorIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(incr::HMMPriorIncrementer, prev::Array{Int,1})
    # NOTE: the interface allows us to use the whole history of the particle;
    # we only use the most recent state prev[end]
    rand(Categorical(incr.hmm.transition_model[prev[end],:]))
end

function log_weight(incr::HMMPriorIncrementer, prev::Array{Int,1}, cur::Int)
    log(incr.hmm.observation_model[cur, incr.observation])
end

function HMMPriorSMCScheme(hmm::HiddenMarkovModel, observations::Array{Int,1}, num_particles::Int)
    initializer = HMMPriorInitializer(hmm, observations[1])
    incrementers = Array{Any,1}(length(observations) - 1)
    for i = 2:length(observations)
        incrementers[i-1] = HMMPriorIncrementer(hmm, observations[i])
    end
    NoRejuvenationSMCScheme(initializer, incrementers, num_particles)
end

# -- Helper functions for SMC in HMM using conditonal (optimal) proposals

immutable HMMConditionalInitializer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(init::HMMConditionalInitializer)
    lprior = log(init.hmm.initial_state_prior)
    llikelihood = log(init.hmm.observation_model[:,init.observation])
    ldist = lprior .+ llikelihood
    # p(x_1 | y_1)
    rand(Categorical(exp(ldist - logsumexp(ldist))))
end

function log_weight(init::HMMConditionalInitializer, cur::Int)
    lprior = log(init.hmm.initial_state_prior)
    llikelihood = log(init.hmm.observation_model[:,init.observation])
    dist = lprior .+ llikelihood
    # p(y_1) = sum_{x_1} p(x_1) p(y_1 | x_1)
    logsumexp(dist)
end


immutable HMMConditionalIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(incr::HMMConditionalIncrementer, prev::Array{Int,1})
    # NOTE: the interface allows us to use the whole history of the particle;
    # we only use the most recent state prev[end]
    lprior = log(incr.hmm.transition_model[prev[end],:])
    llikelihood = log(incr.hmm.observation_model[:,incr.observation])
    ldist = lprior .+ llikelihood
    # p(x_t | x_{t-1}, y_t)
    rand(Categorical(exp(ldist - logsumexp(ldist))))
end

function log_weight(incr::HMMConditionalIncrementer, prev::Array{Int,1}, cur::Int)
    lprior = log(incr.hmm.transition_model[prev[end],:])
    llikelihood = log(incr.hmm.observation_model[:,incr.observation])
    ldist = lprior .+ llikelihood
    # p(y_t | x_{t-1}) = sum_{x_t} p(x_t | x_{t-1}) p(y_t | x_t)
    logsumexp(ldist)
end

function HMMConditionalSMCScheme(hmm::HiddenMarkovModel, observations::Array{Int,1}, num_particles::Int)
    initializer = HMMConditionalInitializer(hmm, observations[1])
    incrementers = Array{Any,1}(length(observations) - 1)
    for i = 2:length(observations)
        incrementers[i-1] = HMMConditionalIncrementer(hmm, observations[i])
    end
    NoRejuvenationSMCScheme(initializer, incrementers, num_particles)
end
