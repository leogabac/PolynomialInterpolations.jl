#= ==========================================================================================
=============================================================================================

----------------------------------------- TO DO LIST ----------------------------------------

- Gradientes                                                                             DONE
    Implementar funciones para encontrar el gradiente de forma analítica
        grad(model::OneVarPolynomial, x::Real)::Vector
        grad(model::TwoVarPolynomial, x::Vector)::Vector
        grad(model::NVarPolynomial, x::Vector)::Vector
    y una función para evaluar el gradiente en algún punto 
        evalgrad(G, x)

- Hessiana                                                                              DONE
    Implementar funciones para encontrar el hessiana de forma analítica
        hess(model::OneVarPolynomial, x::Real)::Matrix
        hess(model::TwoVarPolynomial, x::Vector)::Matrix
        hess(model::NVarPolynomial, x::Vector)::Matrix
    y una función para evaluar el gradiente en algún punto 
        evalhess(H, x)

- DataLoaders                                                                           DONE
    Es necesario hacer una función
        DataLoader(data::Matrix)::OneVarPolydata
        DataLoader(data::Matrix)::TwoVarPolydata
        DataLoader(data::Matrix)::NVarPolydata
    para no estar usando sortslices(). Además, con ello hay que modificar las
    funciones de interpolate1, interpolate2, e interpolaten y usar multiple dispatch:
        interpolate(data::OneVarPolydata)::OneVarPolynomial
        interpolate(data::TwoVarPolydata)::TwoVarPolynomial
        interpolate(data::NVarPolydata)::NVarPolynomial
    al hacer esto dedicí limpiar el código y documentarlo

- Laplacian                                                                             DONE
    laplacian(model::OneVarPolynomial, x::Real)::OneVarPolynomial
    laplacian(model::TwoVarPolynomial, x::Vector)::TwoVarPolynomial
    laplacian(model::NVarPolynomial, x::Vector)::NVarPolynomial

- Struct para guardar todas las derivadas vectoriales                                   PENDING
    el chiste es tener a la mano todas las derivadas a fin de poderlas usar sin recalcular,
    algo así:
        struct NVarField{T,N}
            poly::NVarPolynomial{T,N}
            grad::Vector{NVarPolynomial{T,N}}
            hess::Matrix{NVarPolynomial{T,N}}
            lap::NVarPolynomial{T,N}
        end 
    y una función que tome un polinomio y lance esa estructura

=============================================================================================
========================================================================================== =#

#= ==========================================================================================
=============================================================================================
preamble & functions
=============================================================================================
========================================================================================== =#

# activación de environment de David
    # (@v1.8) pkg> generate PolyInterp_environment
    # (@v1.8) pkg> activate PolyInterp_environment
    using Pkg
    Pkg.activate("PolyInterp_environment")

using Base: Tuple
import Base: Iterators.take, Iterators.rest, Base.+, Base.-
using LinearAlgebra: eltype
using Plots: length
using Kronecker, LinearAlgebra, FiscomTools;

include("structs.jl")
include("dataloaders.jl")
include("interpolate.jl")
include("algebra.jl")
include("datatests.jl")
include("differential operators.jl")

#= ==========================================================================================
=============================================================================================
interpolation
=============================================================================================
========================================================================================== =#

#                                   R - one variable
# data is loaded and interpolated
poly = data2() |> dataloader |> interpolate
# the interpolation polynomial is compared to the data using their maximum absolute difference
[evalpoly(poly, row[1:end-1]...) - row[end] for row ∈ eachrow(data2())] .|> abs |> maximum

# graphical comparison
x = range(0, stop = 2π, length = 100)
plot(x, x -> evalpoly(poly, x))
plot!(x, x -> sin(x)) # original function

#                                   R² - two variables
# data is loaded and interpolated
poly = data3() |> dataloader |> interpolate
# the interpolation polynomial is compared to the data using their maximum absolute difference
[evalpoly(poly, row[1:end-1]...) - row[end] for row ∈ eachrow(data3())] .|> abs |> maximum

# graphical comparison
x = y = range(0, stop = 2π, length = 100)
heatmap(x, y, (x, y) -> evalpoly(poly, [x, y]...))
heatmap(x, y, (x, y) -> round(sin(x) + cos(y), digits = 2)) # original function

#                                   Rⁿ - n variables
# data is loaded and interpolated
poly = datan() |> dataloader |> interpolate
# the interpolation polynomial is compared to the data using their maximum absolute difference
[evalpoly(poly, row[1:end-1]...) - row[end] for row ∈ eachrow(datan())] .|> abs |> maximum

#= ==========================================================================================
=============================================================================================
algebra - addition and subtraction
=============================================================================================
========================================================================================== =#

#                                   R - one variable
f = OneVarPolynomial([0,1,2], [1,3,2] ) # f(x) = 1 + 3x + 2x²
g = OneVarPolynomial([0,1,4], [1,5,7] ) # g(x) = 1 + 5x + 7x⁴
f+g # (f+g)(x) = 2 + 8x + 2x² + 7x⁴
f-g # (f-g)(x) = -2x + 2x² - 7x⁴
f-f # 0

#                                   R² - two variables
f = TwoVarPolynomial{Int64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # f(x) = 1 + 2y + 3x + 4xy
g = TwoVarPolynomial{Int64}([(0,0), (0,1), (1,0), (2,1)], [1,3,5,16] ) # g(x) = 1 + 3y + 5x + 16x²y
f+g # (f+g)(x) = 2 + 5y + 8x + 4xy + 16x²y
f-g # (f-g)(x) = -y - 2x + 4xy - 16x²y
f-f # 0

#                                   Rⁿ - n variables
f = NVarPolynomial{Int64,2}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # f(x) = 1 + 2y + 3x + 4xy
g = NVarPolynomial{Int64,2}([(0,0), (0,1), (1,0), (2,1)], [1,3,5,16] ) # g(x) = 1 + 3y + 5x + 16x²y
f+g # (f+g)(x) = 2 + 5y + 8x + 4xy + 16x²y
f-g # (f-g)(x) = -y - 2x + 4xy - 16x²y
f-f # 0

#= ==========================================================================================
=============================================================================================
differential operators
=============================================================================================
========================================================================================== =#

#                                   R - one variable
f = OneVarPolynomial([0,1,2], [1,3,2] ) # f(x) = 1 + 3x + 2x²
grad_f = grad(f) # f'(x) = 3 + 4x 
evalgrad(grad_f, [1]) # f'(1) = 7
hess_f = hess(f) # f''(x) = 4 
evalhess(hess_f, [1]) # f''(1) = 4
lap_f = laplacian(f) # f''(x) = 4 
evalpoly(lap_f, 1) # f''(1) = 4

#                                   R² - two variables
f = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1), (3,0)], [1,2,3,4,2] ) # 1 + 2y + 3x + 4xy + + 2x³
grad_f = grad(f) # ∇f = [3 + 4y + 6x², 2 + 4x]
evalgrad(grad_f, [1,1]) # ∇f(1,1) = [13, 6]
hess_f = hess(f) # hess(f) = [12x, 4; 4, 0]
evalhess(hess_f, [1,1]) # hess(f)(1,1) = [12, 4; 4, 0]
lap_f = laplacian(f) # ∂²/∂x² f + ∂²/∂y² f = 12x
evalpoly(lap_f, [1,1]...) # (∂²/∂x² f + ∂²/∂y² f)(1,1) = 12

#                                   Rⁿ - n variables - 1
f = NVarPolynomial{Float64,2}([(0,0), (0,1), (1,0), (1,1), (3,0)], [1,2,3,4,2] ) # 1 + 2y + 3x + 4xy + + 2x³
grad_f = grad(f) # ∇f = [3 + 4y + 6x², 2 + 4x]
evalgrad(grad_f, [1,1]) # ∇f(1,1) = [13, 6]
hess_f = hess(f) # hess(f) = [12x, 4; 4, 0]
evalhess(hess_f, [1,1]) # hess(f)(1,1) = [12, 4; 4, 0]
lap_f = laplacian(f) # ∂²/∂x² f + ∂²/∂y² f = 12x
evalpoly(lap_f, [1,1]...) # (∂²/∂x² f + ∂²/∂y² f)(1,1) = 12

#                                   Rⁿ - n variables - 2
f = datan() |> dataloader |> interpolate

grad_f = grad(f) 
evalgrad(grad_f, [1,1,1,1]) 
f_x = diffpoly(f, variable = 1); evalpoly(f_x, [1,1,1,1]...)
f_y = diffpoly(f, variable = 2); evalpoly(f_y, [1,1,1,1]...)
f_z = diffpoly(f, variable = 3); evalpoly(f_z, [1,1,1,1]...)
f_t = diffpoly(f, variable = 4); evalpoly(f_t, [1,1,1,1]...)

hess_f = hess(f)
evalhess(hess_f, [1,1,1,1])
f_xx = diffpoly2(f, variable_1 = 1, variable_2 = 1); evalpoly(f_xx, [1,1,1,1]...)
f_xy = diffpoly2(f, variable_1 = 1, variable_2 = 2); evalpoly(f_xy, [1,1,1,1]...)
f_xz = diffpoly2(f, variable_1 = 1, variable_2 = 3); evalpoly(f_xz, [1,1,1,1]...)
f_xt = diffpoly2(f, variable_1 = 1, variable_2 = 4); evalpoly(f_xt, [1,1,1,1]...)
f_yx = diffpoly2(f, variable_1 = 2, variable_2 = 1); evalpoly(f_yx, [1,1,1,1]...)
f_yy = diffpoly2(f, variable_1 = 2, variable_2 = 2); evalpoly(f_yy, [1,1,1,1]...)
f_yz = diffpoly2(f, variable_1 = 2, variable_2 = 3); evalpoly(f_yz, [1,1,1,1]...)
f_yt = diffpoly2(f, variable_1 = 2, variable_2 = 4); evalpoly(f_yt, [1,1,1,1]...)
f_zx = diffpoly2(f, variable_1 = 3, variable_2 = 1); evalpoly(f_zx, [1,1,1,1]...)
f_zy = diffpoly2(f, variable_1 = 3, variable_2 = 2); evalpoly(f_zy, [1,1,1,1]...)
f_zz = diffpoly2(f, variable_1 = 3, variable_2 = 3); evalpoly(f_zz, [1,1,1,1]...)
f_zt = diffpoly2(f, variable_1 = 3, variable_2 = 4); evalpoly(f_zt, [1,1,1,1]...)
f_tx = diffpoly2(f, variable_1 = 4, variable_2 = 1); evalpoly(f_tx, [1,1,1,1]...)
f_ty = diffpoly2(f, variable_1 = 4, variable_2 = 2); evalpoly(f_ty, [1,1,1,1]...)
f_tz = diffpoly2(f, variable_1 = 4, variable_2 = 3); evalpoly(f_tz, [1,1,1,1]...)
f_tt = diffpoly2(f, variable_1 = 4, variable_2 = 4); evalpoly(f_tt, [1,1,1,1]...)
