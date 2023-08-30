
#struct OneVarPolynomial{T}
#    pows::Vector{Int64}
#    coefficients::Vector{T}
#end

include("vandermonde.jl")

function interpolate1(data::Matrix)
    V = vandermonde( data[:,1] )
    coefficients = ( V\data[:,2] )
    power = [ k for k in 0:(length(coefficients)-1) ]
    return OneVarPolynomial{eltype(coefficients)}( power , coefficients)
end

function interpolate1(x::Vector,y::Vector)
    V = vandermonde(x)
    coefficients = V\y
    power = [ k for k in 0:(length(coefficients)-1) ]
    return OneVarPolynomial{eltype(coefficients)}( power, coefficients )
end

function Base.evalpoly(p::OneVarPolynomial, x::Real)
    sum( p.coefficients .* (x .^ p.pows) )
end

function diffpoly(p::OneVarPolynomial)
    newpows = [current-1  for current in p.pows if current != 0]
    aux_coef = p.pows .* p.coefficients # scale the coefficients properly
    # I opted for this, bc, what if the polynomial by some reason had a zero in its constants
    # we would be deleting them by accident and everything would go brrrr i.e. dimension mismatch
    nonzeropows = findall( x-> x .!= 0, p.pows) # indices of all nonzero powers
    nonzero = aux_coef[nonzeropows]    # i am only interested in the coefficients from powers that are non zero
    # since all these powers, won't vanish in the differentiation process.
    # this will help me to form a new set of powers and coefficients that survive the process of differentiation
    # then only the set of powers and respective coefficients, survive in correspondance,
    # this will make easier "custom polynomials" that could not have all the terms in order.
    @assert length(nonzero) == length(newpows)
    return OneVarPolynomial{eltype(nonzero)}(newpows, nonzero)
end


function intpoly(p::OneVarPolynomial; lims::Vector)
    newpows = [current+1 for current in p.pows]
    newcoef = p.coefficients ./ newpows
    antiderivative = OneVarPolynomial{eltype(newcoef)}(newpows, newcoef)
    return evalpoly(antiderivative, lims[2]) - evalpoly(antiderivative, lims[1]) 
end
