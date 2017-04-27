function logsumexp(logx::Array{Float64,1})
    maxlog = maximum(logx)
    maxlog + log(sum(exp(logx - maxlog)))
end

function logsumexp(logx::Array{Float64,2}, dim::Int)
    maxlog = maximum(logx, dim)
    maxlog + log(sum(exp(logx .- maxlog), dim))
end
