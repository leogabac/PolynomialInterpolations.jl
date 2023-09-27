include("vandermonde.jl")

#= ==========================================================================================
=============================================================================================
auxilary functions
=============================================================================================
========================================================================================== =#

# alch no pude documentar bien esta parte

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

function multikron_inv(uniques_x::Vector)
    # creates the inverse of the kronecker products
    #  ( V1 ⊗ V2 ⊗ ⋯ ⊗ VN )^-1 = inv(V1) ⊗ inv(V2) ⊗ ⋯ ⊗ inv(VN) 
    reduce(Kronecker.kron, LinearAlgebra.inv.( vandermonde.(uniques_x) ) )::Matrix{eltype(eltype(uniques_x))}
end

#= ==========================================================================================
=============================================================================================
interpolation algorithms
=============================================================================================
========================================================================================== =#

function interpolate(data::OneVarPolydata)
    # the Vandermode matrix is found
    V = vandermonde( data.x )
    # V(x) c = y is solved 
    coefficients = ( V\data.y )
    # exponents of x in the interpolation polynomial are found
    power = [ k for k in 0:(length(coefficients)-1) ]
    # the interpolation polynomial is returned
    return OneVarPolynomial{eltype(coefficients)}( power , coefficients)
end

function interpolate(data::TwoVarPolydata)
    # maximum orders in the interpolation polynomial for each independent variable is found
    orders = (length(data.x1)-1, length(data.x2)-1 )
    # the range of exponents ofr each independent variable is found
    ranges = ( 0:m for m in reverse(orders) )
    # exponents of all variables in each monomial in the interpolation polynomial are found
    powers = vec( reverse.( collect( Iterators.product(ranges...) ) ) )
    # V(x)⊗V(y) c = z is solved
    coef = Kronecker.kron( LinearAlgebra.inv( vandermonde(data.x1) ), LinearAlgebra.inv( vandermonde(data.x2) ) ) * data.y ;
    # the interpolation polynomial is returned
    return TwoVarPolynomial{eltype(coef)}(powers, coef)
end

function interpolate(data::NVarPolydata)
    # the esotheric thing in the middle creates a generator that contain all ranges for the cartesian product of powers
    # (0:m1, 0:m2, 0:m3,...m,0:mn) where the order mi is the length of the unique minus 1 (as in 1D)
    ranges = gen2tuple( (0:(length(u)-1) for u in reverse(data.x) ), length(data.x) ); # creates a list of ranges
    # ! TYPE UNSTABLE 
    
    # generate the humongous list of all possible combinations of powers
    powers = ranges2vec(ranges, length(ranges))
    # ! TYPE UNSTABLE 

    # V(x₁)⊗V(x₂)⊗...⊗V(xₙ) c = z is solved
    coef = multikron_inv(data.x) * data.y
    # the interpolation polynomial is returned
    return NVarPolynomial{eltype(coef),length(data.x)}( powers , coef )
end