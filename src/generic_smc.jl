import Distributions.Categorical

function resample(log_weights::Array{Float64,1}, n::Int)
    dist = exp.(log_weights - logsumexp(log_weights))
    rand(Categorical(dist), n)
end

immutable SMCScheme
    initializer
    incrementers::Array
    num_particles::Int
end

function smc(scheme::SMCScheme)
    num_steps = length(scheme.incrementers) + 1
    num_particles = scheme.num_particles
    log_weights = Array{Float64,1}(num_particles)
    log_ml_estimate::Float64 = 0.0

    # a particle is an array of components of arbitrary type
    particles = Array{Any,1}(num_particles)
    new_particles = Array{Any,1}(num_particles)

    # initialize particles and log_weights
    for i=1:num_particles
        particles[i], log_weights[i] = forward(scheme.initializer)
    end
    log_ml_estimate += (logsumexp(log_weights) - log(num_particles))

    # iterate through steps
    for t=2:num_steps
        parents = resample(log_weights, num_particles)
        for i=1:num_particles
            parent_particle = particles[parents[i]]
            x, l = forward(scheme.incrementers[t-1], particles[parents[i]])
            new_particles[i], log_weights[i] = (x, l)
        end
        log_ml_estimate += (logsumexp(log_weights) - log(num_particles))
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end

    # pick output particle
    output_index = resample(log_weights, 1)[1]
    output = particles[output_index]
    (output, log_ml_estimate)
end

function conditional_smc(scheme::SMCScheme, output::Any)
    num_steps = length(scheme.incrementers) + 1
    num_particles = scheme.num_particles
    log_weights = Array{Float64,1}(num_particles)
    log_ml_estimate::Float64 = 0.0

    # a particle is an array of components of arbitrary type
    particles = Array{Any,1}(num_particles)
    new_particles = Array{Any,1}(num_particles)

    # select indices of ancestors of the output particle
    ancestry = rand(Categorical(num_particles), num_steps)

    # sample values for the ancestors of the output particle
    ancest_particles = Array{Any,1}(num_steps)
    ancest_particles[end] = output
    ancest_log_weights = Array{Float64,1}(num_steps);
    for t=num_steps:-1:2
        x, l = backward(scheme.incrementers[t-1], ancest_particles[t])
        ancest_particles[t-1], ancest_log_weights[t] = (x, l)
    end
    ancest_log_weights[1] = backward(scheme.initializer, ancest_particles[1])

    # initialize particles and log_weights
    for i=1:num_particles
        if ancestry[1] == i
            x, l = (ancest_particles[1], ancest_log_weights[1])
        else
            x, l = forward(scheme.initializer)
        end
        particles[i], log_weights[i] = (x, l)
    end
    log_ml_estimate += (logsumexp(log_weights) - log(num_particles))

    # iterate through steps
    for t=2:num_steps
        parents = resample(log_weights, num_particles)
        for i=1:num_particles
            if ancestry[t] == i
                x, l = ancest_particles[t], ancest_log_weights[t]
            else
                parent_particle = particles[parents[i]]
                x, l = forward(scheme.incrementers[t-1], parent_particle)
            end
            new_particles[i], log_weights[i] = (x, l)
        end
        log_ml_estimate += (logsumexp(log_weights) - log(num_particles))
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end
    log_ml_estimate
end
