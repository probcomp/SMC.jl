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
