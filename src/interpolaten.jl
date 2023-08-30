
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

function Base.evalpoly(p::NVarPolynomial, x::Real...)
    pure = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    return sum( p.coefficients .* pure   )
end

