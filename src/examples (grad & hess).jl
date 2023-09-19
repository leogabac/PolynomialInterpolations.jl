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
import Base: Iterators.take, Iterators.rest
using LinearAlgebra: eltype
using Plots: length
using Kronecker, LinearAlgebra, FiscomTools;

include("structs.jl")
include("dataloaders.jl")
include("interpolate1.jl")
include("interpolate2.jl")
include("interpolaten.jl")
include("gradient.jl")
include("hessian.jl")

#= ==========================================================================================
=============================================================================================
1D
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos

p1 = OneVarPolynomial([0,1,2], [1,3,2] ) # f(x) = 1 + 3x + 2x²
grad(p1, 1) # f'(x) = 3 + 4x => f'(1) = 7
hess(p1, 1) # f''(x) = 4 => f''(1) = 4

#= ==========================================================================================
=============================================================================================
2D
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos, además comparo con el método numérico explícito y hecho paso a paso

p2 = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy

grad(p2, [1,1]) # ∇f = [3 + 4y, 2 + 4x] => ∇f([1,1]) = [7, 6]
p2x = diffpoly(p2, variable = 1); evalpoly(p2x, [1,1]...)
p2y = diffpoly(p2, variable = 2); evalpoly(p2y, [1,1]...)

hess(p2, [1,1]) # hess(f) = [0, 4; 4, 0]

p2xx = diffpoly2(p2, variable_1 = 1, variable_2 = 1); evalpoly(p2xx, [1,1]...)
p2xy = diffpoly2(p2, variable_1 = 1, variable_2 = 2); evalpoly(p2xy, [1,1]...)
p2yx = diffpoly2(p2, variable_1 = 2, variable_2 = 1); evalpoly(p2yx, [1,1]...)
p2yy = diffpoly2(p2, variable_1 = 2, variable_2 = 2); evalpoly(p2yy, [1,1]...)

#= ==========================================================================================
=============================================================================================
ND - 1
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos, además comparo con el método numérico explícito y hecho paso a paso

pn_1 = NVarPolynomial{Float64,2}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy
grad(pn_1, [1,1])
hess(pn_1, [1,1])

#= ==========================================================================================
=============================================================================================
ND - 2
=============================================================================================
========================================================================================== =#

# para meter algo más complicadito, ahora sí usaré rutinas de interpolación; la función que trabajaré es 
# f(x,y,z,t) = 5 sin(x) cos(y) exp(z) (sin(t + pi/3) + 0.5).
# Acá solo comparé con el resultado numérico hecho paso a paso, pero se puede obtener todo explícitamente con mathematica.

function datan()
    x = collect(Float64, 1:4);
    y = collect(Float64, 1:4);
    z = collect(Float64, -1:3);
    t = collect(Float64, 0:3);

    m = length(x); n = length(y); p = length(z); q = length(t);
    data = Array{Float64}(undef, m * n * p * q, 5);
    c = 1; 
    for i in 1:m
        for j in 1:n
            for k in 1:p
                for l in 1:q
                    data[c,:] = [x[i], y[j], z[k], t[l], round(sin(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
                    c += 1;
                end
            end
        end 
    end
    return data
end

data = sortslices(datan(), dims=1);

pn_2 = interpolaten(data)

grad(pn_2, [1,1,1,1])
pn_2x = diffpoly(pn_2, variable = 1); evalpoly(pn_2x, [1,1,1,1]...)
pn_2y = diffpoly(pn_2, variable = 2); evalpoly(pn_2y, [1,1,1,1]...)
pn_2z = diffpoly(pn_2, variable = 3); evalpoly(pn_2z, [1,1,1,1]...)
pn_2t = diffpoly(pn_2, variable = 4); evalpoly(pn_2t, [1,1,1,1]...)

hess(pn_2, [1,1,1,1])
pn_2xx = diffpoly2(pn_2, variable_1 = 1, variable_2 = 1); evalpoly(pn_2xx, [1,1,1,1]...)
pn_2xy = diffpoly2(pn_2, variable_1 = 1, variable_2 = 2); evalpoly(pn_2xy, [1,1,1,1]...)
pn_2xz = diffpoly2(pn_2, variable_1 = 1, variable_2 = 3); evalpoly(pn_2xz, [1,1,1,1]...)
pn_2xt = diffpoly2(pn_2, variable_1 = 1, variable_2 = 4); evalpoly(pn_2xt, [1,1,1,1]...)
pn_2yx = diffpoly2(pn_2, variable_1 = 2, variable_2 = 1); evalpoly(pn_2yx, [1,1,1,1]...)
pn_2yy = diffpoly2(pn_2, variable_1 = 2, variable_2 = 2); evalpoly(pn_2yy, [1,1,1,1]...)
pn_2yz = diffpoly2(pn_2, variable_1 = 2, variable_2 = 3); evalpoly(pn_2yz, [1,1,1,1]...)
pn_2yt = diffpoly2(pn_2, variable_1 = 2, variable_2 = 4); evalpoly(pn_2yt, [1,1,1,1]...)
pn_2zx = diffpoly2(pn_2, variable_1 = 3, variable_2 = 1); evalpoly(pn_2zx, [1,1,1,1]...)
pn_2zy = diffpoly2(pn_2, variable_1 = 3, variable_2 = 2); evalpoly(pn_2zy, [1,1,1,1]...)
pn_2zz = diffpoly2(pn_2, variable_1 = 3, variable_2 = 3); evalpoly(pn_2zz, [1,1,1,1]...)
pn_2zt = diffpoly2(pn_2, variable_1 = 3, variable_2 = 4); evalpoly(pn_2zt, [1,1,1,1]...)
pn_2tx = diffpoly2(pn_2, variable_1 = 4, variable_2 = 1); evalpoly(pn_2tx, [1,1,1,1]...)
pn_2ty = diffpoly2(pn_2, variable_1 = 4, variable_2 = 2); evalpoly(pn_2ty, [1,1,1,1]...)
pn_2tz = diffpoly2(pn_2, variable_1 = 4, variable_2 = 3); evalpoly(pn_2tz, [1,1,1,1]...)
pn_2tt = diffpoly2(pn_2, variable_1 = 4, variable_2 = 4); evalpoly(pn_2tt, [1,1,1,1]...)