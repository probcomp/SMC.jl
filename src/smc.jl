using Distributions

function resample(log_weights::Array{Float64,1}, n::Int)
    # is there more numerically accurate way?    
    dist = exp(log_weights - logsumexp(log_weights))
    rand(Categorical(dist), n)
end

immutable StateSpaceSMCScheme
    initializer
    incrementers::Array
    num_particles::Int
end

function no_rejuvenation_smc(scheme::StateSpaceSMCScheme)

    num_steps = length(scheme.incrementers) + 1
    num_particles = scheme.num_particles
    log_weights = Array{Float64,1}(num_particles)
    log_ml_estimate::Float64 = 0.0

    # a particle is an array of components of arbitrary type
    particles = Array{Array,1}(num_particles)
    new_particles = Array{Array,1}(num_particles) 

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
            new_component = sample(scheme.incrementers[t-1], particle)
            log_weights[i] = log_weight(scheme.incrementers[t-1], 
                                        particle, new_component)
            new_particles[i] = vcat(particle, [new_component]) 
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

function no_rejuvenation_conditional_smc(scheme::StateSpaceSMCScheme, 
                                         output_particle::Array)
    num_steps = length(scheme.incrementers) + 1
    num_particles = scheme.num_particles
    log_weights = Array{Float64,1}(num_particles)
    log_ml_estimate::Float64 = 0.0

    # a particle is an array of components of arbitrary type
    particles = Array{Array,1}(num_particles)
    new_particles = Array{Array,1}(num_particles)

    # select indices of ancestors of the output particle
    ancestry = rand(Categorical(num_particles), num_steps)

    # initialize particles and log_weights
    for i=1:num_particles
        if ancestry[1] == i
            x = output_particle[1]
        else
            x = sample(scheme.initializer)
        end
        log_weights[i] = log_weight(scheme.initializer, x)
        particles[i] = Array([x])
    end
    log_ml_estimate += (logsumexp(log_weights) - log(num_particles))

    # iterate through steps
    for t=2:num_steps
        parents = resample(log_weights, num_particles)
        for i=1:num_particles
            if ancestry[t] == i
                particle = particles[ancestry[t-1]]
                new_component = output_particle[t]
            else
                particle = particles[parents[i]]
                new_component = sample(scheme.incrementers[t-1], particle)
            end
            log_weights[i] = log_weight(scheme.incrementers[t-1], 
                                        particle, new_component)
            new_particles[i] = vcat(particle, [new_component])
        end
        log_ml_estimate += (logsumexp(log_weights) - log(num_particles))
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end
    log_ml_estimate
end


