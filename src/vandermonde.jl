function vandermonde(x::Vector)
    composite = [ x.^k for k in 0:(length(x)-1)  ]
    return hcat(composite...)::Matrix{eltype(eltype(composite))}
end
