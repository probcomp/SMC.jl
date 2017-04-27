using Base.Test

@testset "logsumexp tests" begin
    include("mathutils.jl")

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
    include("hmm.jl")
    
    initial_state_prior = [0.4, 0.6]
    transition_model = [0.7 0.3; 0.1 0.9]
    observation_model = [0.6 0.3 0.1; 0.2 0.3 0.5]
    hmm = HiddenMarkovModel(initial_state_prior, transition_model, observation_model)
    observations = [1, 2, 3, 3]

    @testset "forward probabilities" begin
        @test isapprox(forward_pass(hmm, observations), exp(forward_pass_log_space(hmm, observations)))

    end

end
