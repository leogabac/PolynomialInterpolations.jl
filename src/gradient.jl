#=
    grad(p)ᵢ = ∂ᵢp,  où p est un champ scalaire
=#

#= ==========================================================================================
=============================================================================================
gradiente
=============================================================================================
========================================================================================== =#

function grad(p::OneVarPolynomial, x::Real)
    # tomo la primera derivada y evalúo en x
    return (diffpoly(p) |> p -> evalpoly(p, x))::Real
end

function grad(p::TwoVarPolynomial, x::Vector)
    # calculo df/dx, y df/dy, guardando todo en un vector
    return [diffpoly(p, variable = 1) |> p -> evalpoly(p, x...), 
            diffpoly(p, variable = 2) |> p -> evalpoly(p, x...)]::Vector
end

function grad(p::NVarPolynomial, x::Vector)
    # encuentro la cantidad de variables en el polinomio
    nvars = length(p.pows[1]);
    # calculo df/dx_i para cada variable y voy guardando en un vector
    return [diffpoly(p, variable = i) |> p -> evalpoly(p, x...) for i in 1:nvars]::Vector
end