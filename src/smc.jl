using Distributions

# TODO shift to using log-weights everywhere

function resample(weights::Array{Float64,1}, n::Int)
    rand(Categorical(weights / sum(weights)), n)
end

immutable GeneralSMCScheme
    initializer # initial kernel
    incrementers::Array # (k,l,weight) packages
    num_particles::Int
end

function general_smc(scheme::GeneralSMCScheme)
    num_steps = length(scheme.incrementers)
    num_particles = scheme.num_particles
    particles = Array{Any,2}(num_steps, num_particles)
    weights = Array{Float64,2}(num_steps, num_particles) # actually in log space
    # initialize first step particles and weights
    for i=1:num_particles
        x = sample(scheme.initializer)
        weights[1,i] = weight(scheme.initializer, x)
        particles[1,i] = x
    end
    # iterate through steps
    for t=2:num_steps
        parents = resample(weights[t-1,:], num_particles)
        for i=1:num_particles
            prev_x = particles[t-1,parents[i]]
            new_x = sample(scheme.incrementers[t-1], prev_x)
            weights[t,i] = weight(scheme.incrementers[t-1], prev_x, new_x)
            particles[t,i] = new_x
        end
    end
    # final output particle
    output_index = resample(weights[num_steps], 1)
end

immutable StateSpaceSMCScheme
    initializer
    incrementers::Array
    num_particles::Int
end

function state_space_smc(scheme::StateSpaceSMCScheme)
    num_steps = length(scheme.incrementers)
    num_particles = scheme.num_particles
    particles = Array{Any,1}(num_particles)
    new_particles = Array{Any,1}(num_particles) # temporary storage
    weights = Array{Float64,1}(num_particles)

    ml_estimate::Float64 = 1.0

    # initialize first step particles and weights
    for i=1:num_particles
        x = sample(scheme.initializer)
        weights[i] = weight(scheme.initializer, x)
        particles[i] = x
    end
    ml_estimate *= mean(weights)

    # iterate through steps
    for t=2:num_steps
        parents = resample(weights, num_particles)
        for i=1:num_particles
            prev_x = particles[parents[i]]
            new_x = sample(scheme.incrementers[t-1], prev_x)
            # overwrite the old weight with the new weight
            weights[i] = weight(scheme.incrementers[t-1], prev_x, new_x)
            new_particles[i] = new_x
        end
        ml_estimate *= mean(weights)
        tmp = particles
        particles = new_particles
        new_particles = tmp
    end

    # final output particle
    output_index = resample(weights, 1)
    output = particles[output_index]
    (output, ml_estimate)
end
