import Distributions.Categorical

immutable HiddenMarkovModel
    initial_state_prior::Array{Float64,1}
    transition_model::Array{Float64,2}
    observation_model::Array{Float64,2}
    log_initial_state_prior::Array{Float64,1}
    log_transition_model::Array{Float64,2}
    log_observation_model::Array{Float64,2}
    num_states::Int
    num_obs::Int

    function HiddenMarkovModel(initial_state_prior::Array{Float64,1},
                               transition_model::Array{Float64,2}, 
                               observation_model::Array{Float64,2})
        if !isapprox(sum(initial_state_prior), 1.)
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
        log_initial_state_prior = log.(initial_state_prior)
        log_transition_model = log.(transition_model)
        log_observation_model = log.(observation_model)
        new(initial_state_prior, transition_model, observation_model, 
            log_initial_state_prior, log_transition_model, log_observation_model,
            num_states, num_obs)
    end
end

num_states(hmm::HiddenMarkovModel) = hmm.num_states
num_observations(hmm::HiddenMarkovModel) = hmm.num_obs

function hmm_simulate(hmm::HiddenMarkovModel, num_steps::Int)
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

function hmm_forward_pass(hmm::HiddenMarkovModel, observations::Array{Int, 1})
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

function hmm_log_forward_pass(hmm::HiddenMarkovModel, 
                              observations::Array{Int, 1})
    # compute the log forward probabilities
    # log p(x_t, y_{1:t-1}) for t=1,...,T
    # NOTE: we are assuming an observation for each time point
    num_steps = length(observations)
    num_states = hmm.num_states
    lfprobs = Array{Float64,2}(num_steps, num_states)
    lfprobs[1,:] = hmm.log_initial_state_prior
    for t=2:num_steps
        obs = observations[t-1]
        lmessage = lfprobs[t-1,:] .+ hmm.log_observation_model[:,obs]
        # rows are x_{t-1}, cols are x_t; sum over rows
        lproduct = lmessage .+ hmm.log_transition_model
        lfprobs[t,:] = logsumexp(lproduct, 1)
    end
    return lfprobs
end

function hmm_marginal_likelihood(hmm::HiddenMarkovModel,
                                 observations::Array{Int, 1})
    # use the forward probability p(x_T, y_{1:T-1})
    fprobs = hmm_forward_pass(hmm, observations)
    # p(y_{1:T}) = sum_{x_T} p(x_T, y_{1:T-1}) p(y_T | x_T)
    sum(fprobs[end,:] .* hmm.observation_model[:,observations[end]])
end

function hmm_log_marginal_likelihood(hmm::HiddenMarkovModel, 
                                 observations::Array{Int, 1})
    lfprobs = hmm_log_forward_pass(hmm, observations)
    logsumexp(lfprobs[end,:] .+ hmm.log_observation_model[:,observations[end]])
end

function hmm_posterior_sample(hmm::HiddenMarkovModel, 
                              observations::Array{Int, 1})
    num_steps = length(observations)
    lfprobs = hmm_log_forward_pass(hmm, observations)
    ldist = lfprobs[end,:] .+ hmm.log_observation_model[:,observations[end]]
    particle = Array{Int,1}(num_steps)
    particle[end] = rand(Categorical(exp.(ldist - logsumexp(ldist))))
    for t = num_steps-1:-1:1
        ldist = lfprobs[t,:]
        ldist = ldist .+ hmm.log_observation_model[:,observations[t]]
        ldist = ldist .+ hmm.log_transition_model[:,particle[t+1]]
        dist = exp.(ldist - logsumexp(ldist))
        particle[t] = rand(Categorical(dist))
    end
    particle 
end

function hmm_log_joint_probability(hmm::HiddenMarkovModel,
                                   x::Array{Int,1},
                                   observations::Array{Int,1})
    if length(x) != length(observations)
        error("length(x) != length(observations)")
    end
    lj = hmm.log_initial_state_prior[x[1]]
    for t=2:length(x)
        lj += hmm.log_transition_model[x[t-1], x[t]]
    end
    for t=1:length(x)
        lj += hmm.log_observation_model[x[t], observations[t]]
    end
    lj
end

# -- Helper functions for SMC in HMM using prior proposals

immutable HMMPriorInitializer
    hmm::HiddenMarkovModel
    observation::Int # should be the first observation
end

function forward(init::HMMPriorInitializer)
    particle = Array{Int,1}([rand(Categorical(init.hmm.initial_state_prior))])
    log_weight = init.hmm.log_observation_model[particle[1], init.observation]
    (particle, log_weight)
end

function backward(init::HMMPriorInitializer, particle::Array{Int,1})
    @assert length(particle) == 1
    init.hmm.log_observation_model[particle[1], init.observation]
end

immutable HMMPriorIncrementer
    hmm::HiddenMarkovModel
    observation::Int
end

function forward(incr::HMMPriorIncrementer, particle::Array{Int,1})
    prev = particle[end]
    cur = rand(Categorical(incr.hmm.transition_model[prev,:]))
    log_weight = incr.hmm.log_observation_model[cur, incr.observation]
    new_particle = vcat(particle, [cur])
    (new_particle, log_weight)
end

function backward(incr::HMMPriorIncrementer, new_particle::Array{Int,1})
    cur = new_particle[end]
    log_weight = incr.hmm.log_observation_model[cur, incr.observation]
    particle = new_particle[1:end-1]
    (particle, log_weight)
end

function HMMPriorSMCScheme(hmm::HiddenMarkovModel, 
                           observations::Array{Int,1}, 
                           num_particles::Int)
    initializer = HMMPriorInitializer(hmm, observations[1])
    incrementers = Array{Any,1}(length(observations) - 1)
    for i = 2:length(observations)
        incrementers[i-1] = HMMPriorIncrementer(hmm, observations[i])
    end
    SMCScheme(initializer, incrementers, num_particles)
end

# -- Helper functions for SMC in HMM using conditonal (optimal) proposals

immutable HMMConditionalInitializer
    hmm::HiddenMarkovModel
    observation::Int
    dist::Categorical
    log_weight::Float64
    function HMMConditionalInitializer(hmm::HiddenMarkovModel, observation::Int)
        lprior = hmm.log_initial_state_prior
        llikelihood = hmm.log_observation_model[:,observation]
        ldist = lprior .+ llikelihood
        log_weight = logsumexp(ldist)
        # log normalized distribution
        ldist = ldist - log_weight
        dist = Categorical(exp.(ldist))
        new(hmm, observation, dist, log_weight)
    end
end

function forward(init::HMMConditionalInitializer)
    # p(x_1 | y_1)
    particle = Array{Int,1}([rand(init.dist)])
    (particle, init.log_weight)
end

function backward(init::HMMConditionalInitializer, particle::Array{Int,1})
    @assert length(particle) == 1
    # p(y_1) = sum_{x_1} p(x_1) p(y_1 | x_1)
    init.log_weight
end


immutable HMMConditionalIncrementer
    hmm::HiddenMarkovModel
    observation::Int
    llikelihood::Array{Float64,1}
    function HMMConditionalIncrementer(hmm::HiddenMarkovModel, observation::Int)
        llikelihood = hmm.log_observation_model[:,observation]

        new(hmm, observation, llikelihood)
    end
end

function forward(incr::HMMConditionalIncrementer, particle::Array{Int,1})
    cur = particle[end]
    lprior = incr.hmm.log_transition_model[cur,:]
    ldist = lprior .+ incr.llikelihood
    # p(x_t | x_{t-1}, y_t)
    log_weight = logsumexp(ldist)
    new_component = rand(Categorical(exp(ldist - log_weight)))
    new_particle = vcat(particle, [new_component])
    (new_particle, log_weight)
end

function backward(incr::HMMConditionalIncrementer, new_particle::Array{Int,1})
    cur = new_particle[end]
    lprior = incr.hmm.log_transition_model[cur,:]
    ldist = lprior .+ incr.llikelihood
    # p(y_t | x_{t-1}) = sum_{x_t} p(x_t | x_{t-1}) p(y_t | x_t)
    log_weight = logsumexp(ldist)
    particle = new_particle[1:end-1]
    (particle, log_weight)
end

function HMMConditionalSMCScheme(hmm::HiddenMarkovModel, 
                                 observations::Array{Int,1}, 
                                 num_particles::Int)
    initializer = HMMConditionalInitializer(hmm, observations[1])
    incrementers = Array{Any,1}(length(observations) - 1)
    for i = 2:length(observations)
        incrementers[i-1] = HMMConditionalIncrementer(hmm, observations[i])
    end
    SMCScheme(initializer, incrementers, num_particles)
end
