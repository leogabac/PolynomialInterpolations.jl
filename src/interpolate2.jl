
#struct TwoVarPolynomial{T}
#    pows::Vector{Tuple{Int64,Int64}}
#    coefficients::Vector{T}
#end

include("vandermonde.jl")

function interpolate2(data::Matrix)
    x,y = unique( data[:,1] ), unique( data[:,2] );
    orders = (length(x)-1, length(y)-1 );
    ranges = ( 0:m for m in reverse(orders) );
    powers = vec( reverse.( collect( Iterators.product(ranges...) ) ) )
    coef = Kronecker.kron( inv( vandermonde(x) ), LinearAlgebra.inv( vandermonde(y) ) ) * data[:,3] ;
    return TwoVarPolynomial{eltype(coef)}(powers, coef)
end

# z component must be ordered accordingly
function interpolate2(x::Vector, y::Vector, z::Vector)
    orders = (length(x)-1, length(y)-1 );
    ranges = ( 0:m for m in reverse(orders) );
    powers = vec( reverse.( collect( Iterators.product(ranges...) ) ) )
    coef = Kronecker.kron( inv( vandermonde(x) ), LinearAlgebra.inv( vandermonde(y) ) ) * z
    return TwoVarPolynomial{eltype(coef)}(powers, coef)
end

function Base.evalpoly(p::TwoVarPolynomial,x::Real...)
    # pure = vec( [x^j * y^k for k in 0:p.order[1], j in 0:p.order[2] ] ) # pure powers, no coefficients included
    pure = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    return sum( p.coefficients .* pure  )
end

# function diffpoly(p::TwoVarPolynomial; variable::Int64)
#     varpows = [ p.pows[k][variable] for k in 1:length(p.pows) ] # obtain the list of the powers of the differentiated variable
#     newvarpows = [ current-1 for current in varpows if current != 0] # subtract one ...
#     nonzeropows = findall( x-> x .!= 0, varpows) # find all indices of nonzero powers 
#     aux_coef = varpows .* p.coefficients # obtain all new coefficients
#     nonzero = aux_coef[nonzeropows] # retrieve respective coeffients

#     # retieve all powers except for the differentiated one, from nonzero differentiated correspondance
#     auxpows = [ current[1:end .!= variable ][1] for current in p.pows[nonzeropows] ]


#     # from the two bunch (things that i subtracted one) and (things that were held constatt)
#     # introduce a set of powers where I put the new subtracted one, just in the right place
#     newpows = [Tuple(zip(take(auxpows[k],variable-1)...,newvarpows[k],rest(auxpows[k], variable)...)...) for k in 1:length(auxpows)]
#     @assert length(nonzero) == length(newpows)
#     return TwoVarPolynomial{eltype(nonzero)}(newpows, nonzero)
# end

function diffpoly(p::TwoVarPolynomial; variable::Int64)
    # loop through the list of powers, and retrieve a set of the powers in the correct "variable" place
    scaling = [ p.pows[k][variable] for k in 1:length(p.pows) ] # obtain the list of the scaling factors

    # form a (0,0,1,...,0) auxiliar tuple with a one in the correct place
    aux = Tuple(vcat(zeros(Int64,variable-1), 1, zeros(Int64,2-variable)))
    # subtract one from the powers whose value at the "variable" place is not zerom and retrieve all of them
    newvarpows = [ current.-aux for current in p.pows if current[variable] != 0] # subtract one ...

    # the scaling vector has all the OG powers in the correct place, hence we look for all the indices
    # corresponding to non zero powers
    nonzeropows = findall( x-> x .!= 0, scaling) # find all indices of nonzero powers 

    # we retrieve update all coefficients
    aux_coef = scaling .* p.coefficients # obtain all new coefficients

    # and retrieve those that come from the correct, non zero "variable" place power.
    nonzero = aux_coef[nonzeropows] # retrieve respective coeffients

    @assert length(nonzero) == length(newvarpows)
    return TwoVarPolynomial{eltype(nonzero)}(newvarpows, nonzero)
end


# the format of the limits is (xinf,yinf) , (xsup,ysup)
function intpoly(p::TwoVarPolynomial; lims::Vector{Tuple{Int64,Int64}})
    newvarpows = [ current.+1 for current in p.pows] # add one ...
    scaling = [ *(current...) for current in newvarpows ] # obtain the list of the scaling factors
    newcoef = p.coefficients ./ scaling
    antiderivative = TwoVarPolynomial{eltype(newcoef)}(newvarpows, newcoef)
    return evalpoly(antiderivative, lims[2]...) - evalpoly(antiderivative, lims[1]...) 
end

