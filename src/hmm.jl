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
                          observationsobservations::Array{Int, 1})
    # TODO [[ combine with marginal likelihood ? ] -- they probably hvae common comp.
end


immutable HMMPriorInitializer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(init::HMMPriorInitializer)
    rand(Categorical(init.hmm.initial_state_prior))
end

function weight(init::HMMPriorInitializer, cur::Int)
    # the likelihood of the observation given state cur
    pdf(Categorical(init.hmm.observation_model[cur,:]), observation)
end

immutable HMMPriorIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(incr::HMMPriorIncrementer, prev::Int)
    rand(Categorical(incr.hmm.transition_model[prev,:]))
end

function weight(init::HMMPriorIncrementer, prev::Int, cur::Int)
    pdf(Categorical(init.hmm.observation_model[cur,:]), observation)
end

immutable HMMConditionalInitializer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(init::HMMConditionalInitializer)
    prior = init.hmm.initial_state_prior
    likelihood = init.hmm.observation_model[:,init.observation]
    dist = prior .* likelihood
    dist = init.hmm.initial_state_prior .* init.hmm.observation_modelp\
    rand(Categorical(dist / sum(dist)))
end

function weight(init::HMMConditionalInitializer, cur::Int)
    prior = init.hmm.initial_state_prior
    likelihood = init.hmm.observation_model[:,init.observation]
    dist = prior .* likelihood
    sum(dist) # TODO check me 
end

immutable HMMConditionalIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function sample(incr::HMMConditionalIncrementer, prev::Int)
    # TODO
end

function weight(incr::HMMConditionalIncrementer, prev::Int, cur::Int)
    # TODO
end
