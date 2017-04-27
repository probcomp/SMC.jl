
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
                         0.2 0.3 0.5]
    hmm = HiddenMarkovModel(initial_state_prior, 
                            transition_model, 
                            observation_model)
    observations = [1, 3, 2]

    @testset "forward probabilities" begin
        # p(x_t | y_{1:t-1}) for each t = 1,..,T
        actual = forward_pass(hmm, observations)
        # number of time steps x number of states
        @test size(actual) == (3, 2) 

        # check against manually computed values
        expected = begin

            # p(x_2 = 1, y_1 = 1, y_2 = 3) = p(x_1 = 1, x_2 = 1, y_1 = 1, y_2 = 3) + p(x_1 = 2, x_2 = 1, y_1 = 1, y_2 = 3)
            shared_x2_eq_1 = ((0.4 * 0.7 * 0.6 * 0.1) + (0.6 * 0.1 * 0.2 * 0.1))

            # p(x_2 = 2, y_1 = 1, y_2 = 3) = p(x_1 = 1, x_2 = 2, y_1 = 1, y_2 = 3) + p(x_1 = 2, x_2 = 2, y_1 = 1, y_2 = 3)
            shared_x2_eq_2 = ((0.4 * 0.3 * 0.6 * 0.5) + (0.6 * 0.9 * 0.2 * 0.5)) # 

            f11 = 0.4 # p(x_1 = 1)
            f12 = 0.6 # p(x_1 = 2)

            # p(x_2 = 1, y_1 = 1) = p(x_1 = 1, x_2 = 1, y_1 = 1) + p(x_1 = 2, x_2 = 1, y_1 = 1) 
            f21 = (0.4 * 0.7 * 0.6) + (0.6 * 0.1 * 0.2) 

            # p(x_2 = 2, y_1 = 1) = p(x_1 = 1, x_2 = 2, y_1 = 1) + p(x_1 = 2, x_2 = 2, y_1 = 1) 
            f22 = (0.4 * 0.3 * 0.6) + (0.6 * 0.9 * 0.2)

            # p(x_3 = 1, y_1 = 1, y_2 = 3) = p(x_2 = 1, x_3 = 1, y_1 = 1, y_2 = 3) + p(x_2 = 2, x_3 = 1, y_1 = 1, y_2 = 3)
            f31 = shared_x2_eq_1 * 0.7 + shared_x2_eq_2 * 0.1 

            # p(x_3 = 2, y_1 = 1, y_2 = 3) = p(x_2 = 1, x_3 = 2, y_1 = 1, y_2 = 3) + p(x_2 = 2, x_3 = 2, y_1 = 1, y_2 = 3)
            f32 = shared_x2_eq_1 * 0.3 + shared_x2_eq_2 * 0.9 

            [f11 f12; f21 f22; f31 f32]
        end
        @test isapprox(actual, expected)

        # check log space agrees
        @test isapprox(exp(log_forward_pass(hmm, observations)), expected)
    end

    #@testset "marginal likelihoods" begin
        #@test isapprox(marginal_likelihood(hmm, observations),
                       #exp(log_marginal_likelihood(hmm, observations)))
        # TODO check against manually computed value
    #end

end
