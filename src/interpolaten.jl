
#struct NVarPolynomial{T,N}
#    pows::Vector{Tuple{Vararg{Int64,N}}}
#    coefficients::Vector{T}
#end


include("vandermonde.jl")

function gen2tuple(gen::Base.Generator, N::Int64)
    # convert the generator of complete ranges 0:m into tuple format to allow multiple variables
    Tuple(gen)::NTuple{N,UnitRange{Int64}}
end

function ranges2vec(ranges::NTuple, N::Int64)
    # creates the humongous list of all possible powers
    # the product shoud start from the last variable to the first, hence it was reversed in the generator 
    # one line before
    vec( reverse.( collect(Tuple{Vararg{Int64,length(ranges)}},Iterators.product(ranges...) ) ) )::Vector{NTuple{N,Int64}}
end

function multikron(uniques::Vector)
    # creates the inverse of the kronecker products
    #  ( V1 ⊗ V2 ⊗ ⋯ ⊗ VN )^-1 = inv(V1) ⊗ inv(V2) ⊗ ⋯ ⊗ inv(VN) 
    reduce(Kronecker.kron, LinearAlgebra.inv.( vandermonde.(uniques) ) )::Matrix{eltype(eltype(uniques))}
end

# data must be sorted accordingly
function interpolaten(data::Matrix)
    uniques = [unique(data[:,k]) for k in 1:size(data,2)-1] # retrieve unique vectors in data

    # the esotheric thing in the middle creates a generator that contain all ranges for the cartesian product of powers
    # (0:m1, 0:m2, 0:m3,...m,0:mn) where the order mi is the length of the unique minus 1 (as in 1D)
    ranges = gen2tuple( (0:(length(u)-1) for u in reverse(uniques) ), length(uniques) ); # creates a list of ranges
    # ! TYPE UNSTABLE 
    
    # generate the humongous list of all possible combinations of powers
    powers = ranges2vec(ranges, length(ranges))
    # ! TYPE UNSTABLE 

    # compute the inverse matrix
    Ainv = multikron(uniques)
    return NVarPolynomial{eltype(Ainv),length(ranges)}( powers , Ainv*data[:,end] )
end

# según yo tu rutina para TwoVarPolynomial funciona directo para NVarPolynomial; solo generalicé lo de aux y el return
# hice pruebas comparando con resultados analíticos y todo se ve en orden
function diffpoly(p::NVarPolynomial; variable::Int64)
    nvars = length(p.pows[1]);
    # loop through the list of powers, and retrieve a set of the powers in the correct "variable" place
    scaling = [ p.pows[k][variable] for k in 1:length(p.pows) ] # obtain the list of the scaling factors

    # form a (0,0,1,...,0) auxiliar tuple with a one in the correct place
    aux = Tuple(vcat(zeros(Int64,variable-1), 1, zeros(Int64,nvars-variable)))
    # subtract one from the powers whose value at the "variable" place is not zero and retrieve all of them
    newvarpows = [ current.-aux for current in p.pows if current[variable] != 0] # subtract one ...

    # the scaling vector has all the OG powers in the correct place, hence we look for all the indices
    # corresponding to non zero powers
    nonzeropows = findall( x-> x .!= 0, scaling) # find all indices of nonzero powers 

    # we retrieve update all coefficients
    aux_coef = scaling .* p.coefficients # obtain all new coefficients

    # and retrieve those that come from the correct, non zero "variable" place power.
    nonzero = aux_coef[nonzeropows] # retrieve respective coeffients

    @assert length(nonzero) == length(newvarpows)
    return NVarPolynomial{eltype(nonzero), nvars}(newvarpows, nonzero)
end


function Base.evalpoly(p::NVarPolynomial, x::Real...)
    pure = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    return sum( p.coefficients .* pure   )
end