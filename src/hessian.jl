#=
    hess(p)ᵢⱼ = ∂ᵢ∂ⱼp,  où p est un champ scalaire
=#

#= ==========================================================================================
=============================================================================================
segundas derivadas
=============================================================================================
========================================================================================== =#

# no pienso que sea lo más eficiente, pero funciona y permitirá escribir el resto de los códigos. En ambos casos se pipelinean ambas derivadas

function diffpoly2(p::OneVarPolynomial)
    return p |> diffpoly |> diffpoly
end

function diffpoly2(p::Union{TwoVarPolynomial,NVarPolynomial}; variable_1::Int64, variable_2::Int64)
    return p |> p -> diffpoly(p, variable = variable_1) |> p -> diffpoly(p, variable = variable_2)
end

#= ==========================================================================================
=============================================================================================
hessiana
=============================================================================================
========================================================================================== =#

function hess(p::OneVarPolynomial, x::Real)
    # tomo la segunda derivada y evalúo en x
    return (p |> diffpoly2 |> p -> evalpoly(p, x))::Real
end

# en dos o más dimensiones se aprovecha el teorema de Clairaut (d²f/dxdy = d²f/dydx) para ahorrar tiempo

function hess(p::TwoVarPolynomial, x::Vector)
    # calculo d²f/d²x, d²f/dxdy y d²f/d²y, guardando todo en una matriz
    hess_p = [diffpoly2(p, variable_1 = 1, variable_2 = 1) |> p -> evalpoly(p, x...) diffpoly2(p, variable_1 = 1, variable_2 = 2) |> p -> evalpoly(p, x...); 
                    nothing                                                          diffpoly2(p, variable_1 = 2, variable_2 = 2) |> p -> evalpoly(p, x...)]
    
    # lleno los espacios que tienen nothing con el elemento que correspondería a la matriz transpuesta; por el teorema de Clairaut hess_p será simétrica
    return [(!isnothing(hess_p[i,j])) ? hess_p[i,j] : hess_p[j,i] for i in 1:2, j in 1:2]::Matrix
end

function hess(p::NVarPolynomial, x::Vector)
    # encuentro la cantidad de variables en el polinomio
    nvars = length(p.pows[1]);

    # calculo la diagonal y el triángulo superior de la hessiana
    hess_p = [(i<=j) ? diffpoly2(p, variable_1 = i, variable_2 = j) |> p -> evalpoly(p, x...) : nothing for i in 1:nvars, j in 1:nvars]
    
    # lleno los espacios que tienen nothing con el elemento que correspondería a la matriz transpuesta; por el teorema de Clairaut hess_p será simétrica
    return [(!isnothing(hess_p[i,j])) ? hess_p[i,j] : hess_p[j,i] for i in 1:nvars, j in 1:nvars]::Matrix
end