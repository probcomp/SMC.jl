using Distributions

function resample(log_weights::Array{Float64,1}, n::Int)
    # is there more numerically accurate way?    
    dist = exp(log_weights - logsumexp(log_weights))
    rand(Categorical(dist), n)
end

immutable GeneralSMCScheme
    initializer # initial kernel
    incrementers::Array # (k,l,weight) packages
    num_particles::Int
end

function general_smc(scheme::GeneralSMCScheme)
    num_steps = length(scheme.incrementers)
    num_particles = scheme.num_particles

    # TODO we don't need to store all of these particles and log_weights
    # (make it like the state space one, and possibly combine them)
    particles = Array{Any,2}(num_steps, num_particles)
    log_weights = Array{Float64,2}(num_steps, num_particles)

    log_ml_estimate::Float64 = 0.0

    # initialize first step particles and log_weights
    for i=1:num_particles
        x = sample(scheme.initializer)
        log_weights[1,i] = log_weight(scheme.initializer, x)
        particles[1,i] = x
    end
    log_ml_estimate += (logsumexp(log_weights[1,:]) - log(num_particles))

    # iterate through steps
    for t=2:num_steps
        parents = resample(log_weights[t-1,:], num_particles)
        for i=1:num_particles
            prev_x = particles[t-1,parents[i]]
            new_component = sample(scheme.incrementers[t-1], prev_x)
            log_weights[t,i] = log_weight(scheme.incrementers[t-1], prev_x, new_component)
            particles[t,i] = new_component
        end
        log_ml_estimate += (logsumexp(log_weights[t,:]) - log(num_particles))
    end

    output_index = resample(log_weights[end,:], 1)
    output = particles[end,output_index]
    (output, log_ml_estimate)
end

immutable StateSpaceSMCScheme
    initializer
    incrementers::Array
    num_particles::Int
end

function no_rejuvenation_smc(scheme::StateSpaceSMCScheme)
    # a particle is an array of components of arbitrary type
    num_steps = length(scheme.incrementers) + 1
    num_particles = scheme.num_particles
    particles = Array{Array,1}(num_particles)
    new_particles = Array{Array,1}(num_particles) # temporary storage
    log_weights = Array{Float64,1}(num_particles)

    log_ml_estimate::Float64 = 0.0

    # initialize first step particles and log_weights
    for i=1:num_particles
        x = sample(scheme.initializer)
        log_weights[i] = log_weight(scheme.initializer, x)
        particles[i] = Array([x])
    end
    log_ml_estimate += (logsumexp(log_weights) - log(num_particles))

    # iterate through steps
    for t=2:num_steps
        parents = resample(log_weights, num_particles)
        for i=1:num_particles
            particle = particles[parents[i]]
            new_component = sample(scheme.incrementers[t-1], particle) # samples the next component
            # overwrite the old weight with the new weight
            log_weights[i] = log_weight(scheme.incrementers[t-1], particle, new_component)
            new_particles[i] = vcat(particle, [new_component]) # augment particle with new component
        end
        log_ml_estimate += (logsumexp(log_weights) - log(num_particles))
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end
    output_index = resample(log_weights, 1)[1]
    output = particles[output_index]
    @assert length(output) == num_steps
    (output, log_ml_estimate)
end

