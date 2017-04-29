using SMCJulia
using Base.Test

@testset "logsumexp tests" begin

    # on vector
    vec = [1., 2.2, 3.1]
    actual = logsumexp(vec)
    expected = log(sum(exp(vec)))
    @test isapprox(actual, expected)

    # on 2D array
    arr = [[1. 2. 3.]; [3.3 4.1 5.2]]
    # sum over rows
    actual = logsumexp(arr, 1)
    expected = [3.3955454645979626 4.215519523179755 5.305083319768697] 
    @test isapprox(actual, expected)

    # sum over columns
    actual = logsumexp(arr, 2)
    expected = [3.40760596444438 5.593689178496464]'
    @test isapprox(actual, expected)
end

@testset "hmm tests" begin
    
    initial_state_prior = [0.4, 0.6]
    transition_model = [0.7 0.3; 
                        0.1 0.9]
    observation_model = [0.6 0.3 0.1; 
                         0.1 0.4 0.5]
    hmm = HiddenMarkovModel(initial_state_prior, 
                            transition_model, 
                            observation_model)
    observations = [1, 3, 2]

    @testset "forward probabilities" begin
        # p(x_t | y_{1:t-1}) for each t = 1,..,T
        actual = SMCJulia.forward_pass(hmm, observations)
        # number of time steps x number of states
        @test size(actual) == (3, 2) 

        # check against manually computed values
        expected = begin

            # p(x_2 = 1, y_1 = 1, y_2 = 3) = p(x_1 = 1, x_2 = 1, y_1 = 1, y_2 = 3) + p(x_1 = 2, x_2 = 1, y_1 = 1, y_2 = 3)
            shared_x2_eq_1 = ((0.4 * 0.7 * 0.6 * 0.1) + (0.6 * 0.1 * 0.1 * 0.1))

            # p(x_2 = 2, y_1 = 1, y_2 = 3) = p(x_1 = 1, x_2 = 2, y_1 = 1, y_2 = 3) + p(x_1 = 2, x_2 = 2, y_1 = 1, y_2 = 3)
            shared_x2_eq_2 = ((0.4 * 0.3 * 0.6 * 0.5) + (0.6 * 0.9 * 0.1 * 0.5)) # 

            f11 = 0.4 # p(x_1 = 1)
            f12 = 0.6 # p(x_1 = 2)

            # p(x_2 = 1, y_1 = 1) = p(x_1 = 1, x_2 = 1, y_1 = 1) + p(x_1 = 2, x_2 = 1, y_1 = 1) 
            f21 = (0.4 * 0.7 * 0.6) + (0.6 * 0.1 * 0.1) 

            # p(x_2 = 2, y_1 = 1) = p(x_1 = 1, x_2 = 2, y_1 = 1) + p(x_1 = 2, x_2 = 2, y_1 = 1) 
            f22 = (0.4 * 0.3 * 0.6) + (0.6 * 0.9 * 0.1)

            # p(x_3 = 1, y_1 = 1, y_2 = 3) = p(x_2 = 1, x_3 = 1, y_1 = 1, y_2 = 3) + p(x_2 = 2, x_3 = 1, y_1 = 1, y_2 = 3)
            f31 = shared_x2_eq_1 * 0.7 + shared_x2_eq_2 * 0.1 

            # p(x_3 = 2, y_1 = 1, y_2 = 3) = p(x_2 = 1, x_3 = 2, y_1 = 1, y_2 = 3) + p(x_2 = 2, x_3 = 2, y_1 = 1, y_2 = 3)
            f32 = shared_x2_eq_1 * 0.3 + shared_x2_eq_2 * 0.9 

            [f11 f12; f21 f22; f31 f32]
        end
        @test isapprox(actual, expected)

        # check log space agrees
        @test isapprox(exp(SMCJulia.log_forward_pass(hmm, observations)), expected)
    end

    @testset "marginal likelihoods" begin
        actual = SMCJulia.marginal_likelihood(hmm, observations)

        expected = (
              (0.4 * 0.7 * 0.7 * 0.6 * 0.1 * 0.3) + # x = (1, 1, 1)
              (0.4 * 0.7 * 0.3 * 0.6 * 0.1 * 0.4) + # x = (1, 1, 2)
              (0.4 * 0.3 * 0.1 * 0.6 * 0.5 * 0.3) + # x = (1, 2, 1)
              (0.4 * 0.3 * 0.9 * 0.6 * 0.5 * 0.4) + # x = (1, 2, 2)
              (0.6 * 0.1 * 0.7 * 0.1 * 0.1 * 0.3) + # x = (2, 1, 1)
              (0.6 * 0.1 * 0.3 * 0.1 * 0.1 * 0.4) + # x = (2, 1, 2)
              (0.6 * 0.9 * 0.1 * 0.1 * 0.5 * 0.3) + # x = (2, 2, 1)
              (0.6 * 0.9 * 0.9 * 0.1 * 0.5 * 0.4)) # x = (2, 2, 2)
        
        @test isapprox(actual, expected)

        # check that log space agrees
        @test isapprox(actual, exp(log_marginal_likelihood(hmm, observations)))
    end

    @testset "posterior sampling" begin
        particle = posterior_sample(hmm, observations)
        
        # smoke test: check the type and dimensions and other basic properties
        @test typeof(particle) == Array{Int,1}
        @test length(particle) == 3
        @test all(particle .>= 1)
        @test all(particle .<= 2)

        # TODO: test the distribution? Maybe next year :)

    end

    @testset "sanity checks" begin

        # sanity check on extreme observation
        srand(1)
        hmm = HiddenMarkovModel([0.5, 0.5],                     # prior
                                [0.5 0.5; 0.5 0.5],             # transition
                                [0.9999 0.0001; 0.0001 0.9999]) # observation
        obs = [1]
        num_samples = 100
        num_1 = 0
        for i = 1:num_samples
            particle = posterior_sample(hmm, obs)
            num_1 += (particle[1] == 1)
        end
        @test num_1 >= 95

        # sanity check on extreme prior and transition
        srand(1)
        hmm = HiddenMarkovModel([0.0001, 0.9999],                     # prior
                                [0.9999 0.0001; 0.0001 0.9999],             # transition
                                [0.5 0.5; 0.5 0.5]) # observation
        obs = [1, 2]
        num_samples = 100
        num_1 = 0
        for i = 1:num_samples
            particle = posterior_sample(hmm, obs)
            num_1 += (particle[2] == 1)
        end
        @test num_1 <= 5

        # a symmetric extreme distribution
        hmm = HiddenMarkovModel([0.5, 0.5],         # prior
                                [0.9999 0.0001; 0.0001 0.9999],  # transition
                                [0.9999 0.0001; 0.0001 0.9999]) # observation

        # switch happens at step 9 -> 10
        obs_a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        obs_b = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

        # test for symmetry
        @test isapprox(log_marginal_likelihood(hmm, obs_a),
                       log_marginal_likelihood(hmm, obs_b))

        # check samples
        srand(1)
        num_samples = 100

        num_start_1 = 0
        num_end_2 = 0
        for i = 1:num_samples
            particle = posterior_sample(hmm, obs_a)
            num_start_1 += all(particle[1:9] .== 1)
            num_end_2 += all(particle[10:end] .== 2)
        end
        @test num_start_1 >= 95
        @test num_end_2 >= 95

        srand(1)
        num_start_2 = 0
        num_end_1 = 0
        for i = 1:num_samples
            particle = posterior_sample(hmm, obs_b)
            num_start_2 += all(particle[1:9] .== 2)
            num_end_1 += all(particle[10:end] .== 1)
        end
        @test num_start_2 >= 95
        @test num_end_1 >= 95
    end

end

@testset "smc tests" begin

    hmm = HiddenMarkovModel(
            [0.4, 0.6], # prior
            [0.7 0.3; 
             0.1 0.9], # transition
            [0.6 0.3 0.1; 
             0.1 0.4 0.5]) # observation
    observations = [1, 3, 2]

    # SMC with prior proposal
    srand(1)
    num_particles = 10000
    scheme = HMMPriorSMCScheme(hmm, observations, num_particles)
    output, log_ml_estimate = smc(scheme)
    @test length(output) == 3
    expected = log_marginal_likelihood(hmm, observations)
    @test isapprox(log_ml_estimate, expected, atol=0.1, rtol=0)

    # conditional SMC (take output from above as input)
    log_ml_estimate = conditional_smc(scheme, output)
    @test isapprox(log_ml_estimate, expected, atol=0.1, rtol=0)

    # SMC with conditional proposal
    srand(1)
    num_particles = 10000
    scheme = HMMConditionalSMCScheme(hmm, observations, num_particles)
    output, log_ml_estimate = smc(scheme)
    @test length(output) == 3
    expected = log_marginal_likelihood(hmm, observations)
    @test isapprox(log_ml_estimate, expected, atol=0.02, rtol=0)

    # conditional SMC (take output from above as input)
    log_ml_estimate = conditional_smc(scheme, output)
    @test isapprox(log_ml_estimate, expected, atol=0.1, rtol=0)
end

@testset "smc sandwiching the evidence with exact sampling" begin

    hmm = HiddenMarkovModel(
            [0.4, 0.6], # prior
            [0.7 0.3; 
             0.1 0.9], # transition
            [0.6 0.3 0.1; 
             0.1 0.4 0.5]) # observation
    observations = [1, 3, 2]

    # true marginal likelihood
    expected = log_marginal_likelihood(hmm, observations)

    function test_bounds(smc_scheme_constructor, all_num_particles::Array{Int,1},
                         num_reps::Int)
        lower_bound_raw_data = Array{Float64, 2}(length(all_num_particles), num_reps)
        upper_bound_raw_data = Array{Float64, 2}(length(all_num_particles), num_reps)
        for (i, num_particles) in enumerate(all_num_particles)
            scheme = HMMConditionalSMCScheme(hmm, observations, num_particles)
            for j = 1:num_reps
                output, lml_estimate = smc(scheme)
                lower_bound_raw_data[i, j] = lml_estimate 
            end
            for j = 1:num_reps
                output = posterior_sample(hmm, observations)
                lml_estimate = conditional_smc(scheme, output)
                upper_bound_raw_data[i, j] = lml_estimate 
            end
        end
    
        lower_bound_means = mean(lower_bound_raw_data, 2)
        lower_bound_stds = std(lower_bound_raw_data, 2)
        upper_bound_means = mean(upper_bound_raw_data, 2)
        upper_bound_stds = std(upper_bound_raw_data, 2)

        # they should be upper and lower bounds in all cases
        for i=1:length(all_num_particles)
            @test lower_bound_means[i] <= expected
            @test upper_bound_means[i] >= expected
        end

        # the estimates should get tighter and less noisy with more particles
        for i=1:length(all_num_particles)-1
            @test lower_bound_means[i] <= lower_bound_means[i+1]
            @test upper_bound_means[i] >= upper_bound_means[i+1]
            @test lower_bound_stds[i] >= lower_bound_stds[i+1]
            @test upper_bound_stds[i] >= upper_bound_stds[i+1]
        end

        # the final estimates should both be tight
        @test isapprox(lower_bound_means[end], expected, atol=0.01, rtol=0)
        @test isapprox(upper_bound_means[end], expected, atol=0.01, rtol=0)

    end

    # SMC with prior proposal
    srand(1)
    test_bounds(HMMConditionalSMCScheme, [1, 3, 10, 30, 100], 1000)

    # SMC with conditional proposal
    srand(1)
    test_bounds(HMMPriorSMCScheme, [1, 3, 10, 30, 100], 1000)

end
