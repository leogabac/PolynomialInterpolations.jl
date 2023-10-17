#= ==========================================================================================
=============================================================================================
first order derivative

∂ᵢ = ∂/∂xᵢ
=============================================================================================
========================================================================================== =#

function diffpoly(p::OneVarPolynomial)
    # d/dx (a x^n) = an x^(n - 1)

    #                   x^(n - 1) term    
    # the exponents of the differentiated polynomial are found
    new_pows = p.pows[1:end-1]
    # from the original polynomial, only terms with a non-zero power are kept
    index_nonzero_pows = eachindex(p.pows)[2:end]
    #                   an term           
    # the coefficients of the differentiated polynomial are found
    new_coef = (p.pows .* p.coefficients )[index_nonzero_pows]
    # the differentiated polynomial is returned
    return OneVarPolynomial{eltype(new_coef)}(new_pows, new_coef) |> empty2zero_poly
end

function diffpoly(p::TwoVarPolynomial; variable::Int64)
    # d/dx (a x^n y^m) = an x^(n - 1) y^m
    # d/dy (a x^n y^m) = am x^m y^(m - 1)

    #                   x^(n - 1) y^m term    
    # to find the new exponents, an auxilary tuple is built. It will shortly be used to subtract.
    aux = (variable == 1) ? ((1, 0)) : (0, 1)
    # taking only the terms where the differential variable has a non-zero exponent, the auxilary tuple is subtracted from the exponents of all monomials
    new_pows = [ current.-aux for current in p.pows if current[variable] != 0]
    
    #                   n term                
    # loop through the list of powers, and retrieve a set of the powers in the correct "variable" place
    scaling = [ pow[variable] for pow in p.pows ] 
    # from the original polynomial, only terms where the differential variable has a non-zero exponent are kept
    index_nonzero_pows = findall( x-> x .!= 0, scaling)
    #                   an term               
    # the coefficients of the differentiated polynomial are found
    new_coef = (p.coefficients .* scaling)[index_nonzero_pows]
    
    # sanity check
    @assert length(new_coef) == length(new_pows)
    # the differentiated polynomial is returned
    return TwoVarPolynomial{eltype(new_coef)}(new_pows, new_coef) |> empty2zero_poly
end

function diffpoly(p::NVarPolynomial; variable::Int64)
    # d/dxᵢ a x₁^n₁...xᵢ^nᵢ...xₙ^nₙ = anᵢ x₁^n₁...xᵢ^(nᵢ-1)...xₙ^nₙ

    # the number of variables is found
    nvars = length(p.pows[1]);
    
    #                   x₁^n₁...xᵢ^(nᵢ-1)...xₙ^nₙ term    
    # to find the new exponents, an auxilary tuple is built. It will shortly be used to subtract.
    aux = Tuple(vcat(zeros(Int64,variable-1), 1, zeros(Int64,nvars-variable)))
    # taking only the terms where the differential variable has a non-zero exponent, the auxilary tuple is subtracted from the exponents of all monomials
    new_pows = [ current.-aux for current in p.pows if current[variable] != 0]
    
    #                   nᵢ term                           
    # loop through the list of powers, and retrieve a set of the powers in the correct "variable" place
    scaling = [ p.pows[k][variable] for k in 1:length(p.pows) ]
    # from the original polynomial, only terms where the differential variable has a non-zero exponent are kept
    index_nonzero_pows = findall( x-> x .!= 0, scaling)
    #                   anᵢ term                          
    # the coefficients of the differentiated polynomial are found
    new_coef = (p.coefficients .* scaling)[index_nonzero_pows]

    # sanity check
    @assert length(new_coef) == length(new_pows)
    # the differentiated polynomial is returned
    return NVarPolynomial{eltype(new_coef), nvars}(new_pows, new_coef) |> empty2zero_poly
end

#= ==========================================================================================
=============================================================================================
second order derivative

∂ᵢ∂ⱼ = ∂²/∂xᵢ∂xⱼ
=============================================================================================
========================================================================================== =#

# no pienso que sea lo más eficiente, pero funciona y permitirá escribir el resto de los códigos. En ambos casos se pipelinean ambas derivadas

function diffpoly2(p::OneVarPolynomial)
    return (p |> diffpoly |> diffpoly)::OneVarPolynomial
end

function diffpoly2(p::Union{TwoVarPolynomial,NVarPolynomial}; variable_1::Int64, variable_2::Int64)
    return (p |> p -> diffpoly(p, variable = variable_1) |> p -> diffpoly(p, variable = variable_2))::Union{TwoVarPolynomial,NVarPolynomial}
end

#= ==========================================================================================
=============================================================================================
gradient

grad(p)ᵢ = ∂ᵢp,  where p is a scalar field
=============================================================================================
========================================================================================== =#

function gradient(p::OneVarPolynomial)
    # the first derivative is computed
    return [diffpoly(p)]::Vector
end

function gradient(p::TwoVarPolynomial)
    # the first derivatives are computed and saved in a vector
    return [diffpoly(p, variable = 1), 
            diffpoly(p, variable = 2)]::Vector
end

function gradient(p::NVarPolynomial)
    # the number of variables is found
    nvars = length(p.pows[1]);
    # the first derivatives are computed and saved in a vector
    return [diffpoly(p, variable = i) for i in 1:nvars]::Vector
end

#= ==========================================================================================
=============================================================================================
hessian

hess(p)ᵢⱼ = ∂ᵢ∂ⱼp,  where p is a scalar field
=============================================================================================
========================================================================================== =#


function hessian(p::OneVarPolynomial)
    # the second derivative is computed and saved in a matrix (for consistency)
    hessian = Array{OneVarPolynomial}(undef,1,1); hessian[1] = p |> diffpoly2
    return hessian::Matrix
end

# since second derivatives commute, the hessian matrix will be symmetric. This is used to save time.

function hessian(p::TwoVarPolynomial)
    # second derivatives are computed (without redundancy)
    hess_p = [diffpoly2(p, variable_1 = 1, variable_2 = 1) diffpoly2(p, variable_1 = 1, variable_2 = 2);
                    nothing                                diffpoly2(p, variable_1 = 2, variable_2 = 2)]
    
    # the hessian matrix is made symmetric
    return [(!isnothing(hess_p[i,j])) ? hess_p[i,j] : hess_p[j,i] for i in 1:2, j in 1:2]::Matrix
end

function hessian(p::NVarPolynomial)
    # the number of variables is found
    nvars = length(p.pows[1]);

    # second derivatives are computed (without redundancy); in this stage, only an upper triangular matrix is found
    hess_p = [(i<=j) ? diffpoly2(p, variable_1 = i, variable_2 = j) : nothing for i in 1:nvars, j in 1:nvars]
    
    # the hessian matrix is made symmetric
    return [(!isnothing(hess_p[i,j])) ? hess_p[i,j] : hess_p[j,i] for i in 1:nvars, j in 1:nvars]::Matrix
end

#= ==========================================================================================
=============================================================================================
laplacian

lap(p) = ∑ ∂²ᵢ p,  where p is a scalar field
=============================================================================================
========================================================================================== =#

function laplacian(p::OneVarPolynomial)
    # the second derivative is computed
    return (p |> diffpoly2)::OneVarPolynomial
end

function laplacian(p::TwoVarPolynomial)
    # the second derivatives are computed and added
    return (diffpoly2(p, variable_1 = 1, variable_2 = 1) + diffpoly2(p, variable_1 = 2, variable_2 = 2))::TwoVarPolynomial
end

function laplacian(p::NVarPolynomial)
    # the number of variables is found
    nvars = length(p.pows[1]);
    # the second derivatives are computed and added
    return (sum(diffpoly2(p, variable_1 = i, variable_2 = i) for i in 1:nvars))::NVarPolynomial
end

#= ==========================================================================================
=============================================================================================
full differential analysis
=============================================================================================
========================================================================================== =#

function diff_analyze(p::OneVarPolynomial)
    return OneVarPolyNDiff{eltype(p.coefficients)}(p, gradient(p), hessian(p), laplacian(p))
end

function diff_analyze(p::TwoVarPolynomial)
    return TwoVarPolyNDiff{eltype(p.coefficients)}(p, gradient(p), hessian(p), laplacian(p))
end

function diff_analyze(p::NVarPolynomial)
    return NVarPolyNDiff{eltype(p.coefficients), length(p.pows[1])}(p, gradient(p), hessian(p), laplacian(p))
end
