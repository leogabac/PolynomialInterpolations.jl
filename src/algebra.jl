#= ==========================================================================================
=============================================================================================
evaluation of a polynomial
=============================================================================================
========================================================================================== =#

function Base.evalpoly(p::OneVarPolynomial, x::Real)
    # P(x) = ∑ cᵢxⁱ
    sum( p.coefficients .* (x .^ p.pows) )
end

function Base.evalpoly(p::TwoVarPolynomial,x::Real...)
    # P(x, y) = ∑ c_B(ij) xⁱ yʲ

    #                   xⁱyʲ term         
    # monomials, though without coefficients, are evaluated
    monomials = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    # the evaluation is returned
    return sum( p.coefficients .* monomials )
end

function Base.evalpoly(p::NVarPolynomial, x::Real...)
    # P(x₁, ..., xₙ) = ∑ ... ∑ c_Bₙ(i₁,...,iₙ) x₁^i₁ ... xₙ^iₙ

    #                   x₁^i₁ ... xₙ^iₙ term         
    # monomials, though without coefficients, are evaluated
    pure = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    # the evaluation is returned
    return sum( p.coefficients .* pure   )
end

#= ==========================================================================================
=============================================================================================
f(x) = 0

taking a derivative or subtracting polynomials may cancel all terms. To avoid returning 
polynomials with empty fields, the following functions will build the function f(x) = 0 and 
write it with the appropriate structure.
=============================================================================================
========================================================================================== =#

function empty2zero_poly(p::OneVarPolynomial)
    # the exponents and coefficients are retrieved to be able to mutate them
    new_pows = p.pows; new_coef = p.coefficients;
    # if there are no exponents, then there are no terms. In that case f(x) = 0 is built
    if isempty(new_pows)
        new_pows = [0]
        new_coef = [0] .|> eltype(p.coefficients)
    end
    # the resulting polynomial is built
    return OneVarPolynomial{eltype(new_coef)}(new_pows, new_coef)
end

function empty2zero_poly(p::TwoVarPolynomial)
    # the exponents and coefficients are retrieved to be able to mutate them
    new_pows = p.pows; new_coef = p.coefficients;
    # if there are no exponents, then there are no terms. In that case f(x) = 0 is built
    if isempty(new_pows)
        new_pows = [(0,0)]
        new_coef = [0] .|> eltype(p.coefficients)
    end
    # the resulting polynomial is built
    return TwoVarPolynomial{eltype(new_coef)}(new_pows, new_coef)
end

function empty2zero_poly(p::NVarPolynomial)
    # the exponents and coefficients are retrieved to be able to mutate them
    new_pows = p.pows; new_coef = p.coefficients;
    # if there are no exponents, then there are no terms. In that case f(x) = 0 is built
    if isempty(new_pows)
        new_pows = [zeros(Int64, fieldcount(eltype(new_pows))) |> Tuple]
        new_coef = [0] .|> eltype(p.coefficients)
    end
    # the resulting polynomial is built
    return NVarPolynomial{eltype(new_coef), length(new_pows[1])}(new_pows, new_coef)
end


#= ==========================================================================================
=============================================================================================
evaluation of gradient vector or hessian matrix
=============================================================================================
========================================================================================== =#

function evalgrad(G::Vector{T} where T<:Union{OneVarPolynomial,TwoVarPolynomial,NVarPolynomial}, x::Vector)
    return (G .|> p -> evalpoly(p, x...))::Vector
end

function evalhess(H::Matrix{T} where T<:Union{OneVarPolynomial,TwoVarPolynomial,NVarPolynomial}, x::Vector)
    return (H .|> p -> evalpoly(p, x...))::Matrix
end

#= ==========================================================================================
=============================================================================================
addition and subtraction of polynomials
=============================================================================================
========================================================================================== =#

function +(p1::OneVarPolynomial, p2::OneVarPolynomial)
    new_pows = unique(vcat(p1.pows,p2.pows))
    new_coef = Vector{typeof(p1.coefficients[1] + p2.coefficients[1])}(undef,length(new_pows))

    p1_index_new_pows = [findfirst(x -> x == pow, p1.pows) for pow in new_pows]
    p2_index_new_pows = [findfirst(x -> x == pow, p2.pows) for pow in new_pows]

    for c in eachindex(new_coef)
        if !isnothing(p1_index_new_pows[c]) && !isnothing(p2_index_new_pows[c])
            new_coef[c] = p1.coefficients[p1_index_new_pows[c]] + p2.coefficients[p2_index_new_pows[c]]
        else
            try
                new_coef[c] = p1.coefficients[p1_index_new_pows[c]]
            catch
                new_coef[c] = p2.coefficients[p2_index_new_pows[c]]
            end
        end
    end

    index_nonzero_coef = findall( x-> x .!= 0, new_coef)

    return OneVarPolynomial(new_pows[index_nonzero_coef], new_coef[index_nonzero_coef]) |> empty2zero_poly 
end

function +(p1::TwoVarPolynomial, p2::TwoVarPolynomial)
    new_pows = unique(vcat(p1.pows,p2.pows))
    new_coef = Vector{typeof(p1.coefficients[1] + p2.coefficients[1])}(undef,length(new_pows))

    p1_index_new_pows = [findfirst(x -> x == pow, p1.pows) for pow in new_pows]
    p2_index_new_pows = [findfirst(x -> x == pow, p2.pows) for pow in new_pows]

    for c in eachindex(new_coef)
        if !isnothing(p1_index_new_pows[c]) && !isnothing(p2_index_new_pows[c])
            new_coef[c] = p1.coefficients[p1_index_new_pows[c]] + p2.coefficients[p2_index_new_pows[c]]
        else
            try
                new_coef[c] = p1.coefficients[p1_index_new_pows[c]]
            catch
                new_coef[c] = p2.coefficients[p2_index_new_pows[c]]
            end
        end
    end

    index_nonzero_coef = findall( x-> x .!= 0, new_coef)

    return TwoVarPolynomial{eltype(new_coef)}(new_pows[index_nonzero_coef], new_coef[index_nonzero_coef] ) |> empty2zero_poly
end

function +(p1::NVarPolynomial, p2::NVarPolynomial)
    (length(p1.pows[1]) !== length(p2.pows[1])) ? (error("Polynomials do not have the same number of variables!")) : nothing

    new_pows = unique(vcat(p1.pows,p2.pows))
    new_coef = Vector{typeof(p1.coefficients[1] + p2.coefficients[1])}(undef,length(new_pows))

    p1_index_new_pows = [findfirst(x -> x == pow, p1.pows) for pow in new_pows]
    p2_index_new_pows = [findfirst(x -> x == pow, p2.pows) for pow in new_pows]

    for c in eachindex(new_coef)
        if !isnothing(p1_index_new_pows[c]) && !isnothing(p2_index_new_pows[c])
            new_coef[c] = p1.coefficients[p1_index_new_pows[c]] + p2.coefficients[p2_index_new_pows[c]]
        else
            try
                new_coef[c] = p1.coefficients[p1_index_new_pows[c]]
            catch
                new_coef[c] = p2.coefficients[p2_index_new_pows[c]]
            end
        end
    end

    index_nonzero_coef = findall( x-> x .!= 0, new_coef)

    return NVarPolynomial{eltype(new_coef), length(new_pows[1])}(new_pows[index_nonzero_coef], new_coef[index_nonzero_coef] ) |> empty2zero_poly
end

function -(p1::OneVarPolynomial, p2::OneVarPolynomial)
    # a - b = a + (-b)
    minus_p2 = OneVarPolynomial(p2.pows, -p2.coefficients)
    return p1 + minus_p2
end

function -(p1::TwoVarPolynomial, p2::TwoVarPolynomial)
    # a - b = a + (-b)
    minus_p2 = TwoVarPolynomial{eltype(p2.coefficients)}(p2.pows, -p2.coefficients)
    return p1 + minus_p2
end

function -(p1::NVarPolynomial, p2::NVarPolynomial)
    # a - b = a + (-b)
    minus_p2 = NVarPolynomial{eltype(p2.coefficients), length(p2.pows[1])}(p2.pows, -p2.coefficients)
    return p1 + minus_p2
end