#= ==========================================================================================
=============================================================================================

----------------------------------------- TO DO LIST ----------------------------------------

- Implementar gradientes:                                                               DONE
    grad(model::OneVarPolynomial, x::Real)::Real
    grad(model::TwoVarPolynomial, x::Vector)::Vector
    grad(model::NVarPolynomial, x::Vector)::Vector

- Implementar hessiana:                                                                 DONE
    hess(model::OneVarPolynomial, x::Real)::Real
    hess(model::TwoVarPolynomial, x::Vector)::Matrix
    hess(model::NVarPolynomial, x::Vector)::Matrix

- DataLoaders


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


#! DATALOADERS
# ===================================================================
#  TEST
# ===================================================================

include("datatests.jl")

dataloader(test1[:,2], test1[:,1])

dataloader(test2[:,3], 
test2[:,1] ,
test2[:,2] )

dataloader(testn[:,5], 
testn[:,1] ,
testn[:,2] ,
testn[:,3] ,
testn[:,4])


d1 = dataloader(test1)
d2 = dataloader(test2)
dn = dataloader(testn)


# ===================================================================
#  EVALUATION TESTS
# ===================================================================

function data2()
    x = 1.:1:10.; 
    y = round.(sin.(x), digits=2); 
    return  [x y]
end
test2 = data2();

x = test2[:,1]
y = test2[:,2]

interpolate1(test2)
interpolate1(x,y)

p1 = interpolate1(x,y)
approx_og = [ evalpoly(p1, xs) for xs in x ]
sum(approx_og - y)/length(y)

# ===================================================================
#  DIFFERENTIATION TESTS
# ===================================================================

p1 = OneVarPolynomial([0,1,2], [1,3,2] ) # 1 + 3x + 2x²
# 3 + 4x
# 4
p = diffpoly(p1)
p = diffpoly(p)
p = diffpoly2(p1)
evalpoly(p, 1)

grad(p1, 1)
hess(p1, 1)

# ===================================================================
#  INTEGRATION TESTS
# ===================================================================

p1 = OneVarPolynomial([0,1,2], [1,3,2] )

p = intpoly(p1, lims = [-10,1] )



#! iNTERPOLATE 2

function data3()
    x = collect(Float64, -0.5:0.5:1.5);
    y = collect(Float64, -0.5:0.5:1.5);
    m = length(x); n = length(y);
    c = 1;
    data = Array{Float64}(undef, m * n, 3);
    dx  = 2.5;
    dy = -0.4;
    for i in 1:m
        for j in 1:n
            data[c,:] = [ x[i], y[j], sin(x[i]) + 2*cos(y[j]) + sin(5*x[i]*y[j])  ]';
            c += 1;
        end 
    end
    return data
end



# ===================================================================
#  TEST
# ===================================================================

data = data3()
# @benchmark sortslices(data, dims = 1);
data = sortslices(data, dims = 1)
p1 = interpolate2(data)

approxog = [ evalpoly(p1,p...) for p in zip(data[:,1],data[:,2]) ]
sum( abs.(approxog - data[:,3]) ) / length(data[:,3])

evalpoly(p1, [1,2]...)
[1,1] isa Vector
@time interpolate2(data); 


# ===================================================================
#  DIFFERENTIATION TEST
# ===================================================================

p1 = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy
# p = diffpoly(p1, variable = 2)
# p = diffpoly(p, variable = 2)
# p = diffpoly2(p1, variable = 2)
# @code_warntype diffpoly(p1, variable = 2)

grad(p1, [1,1])
p1x = diffpoly(p1, variable = 1); evalpoly(p1x, [1,1]...)
p1y = diffpoly(p1, variable = 2); evalpoly(p1y, [1,1]...)

hess(p1, [1,1])

p1xx = diffpoly2(p1, variable_1 = 1, variable_2 = 1); evalpoly(p1xx, [1,1]...)
p1xy = diffpoly2(p1, variable_1 = 1, variable_2 = 2); evalpoly(p1xy, [1,1]...)
p1yx = diffpoly2(p1, variable_1 = 2, variable_2 = 1); evalpoly(p1yx, [1,1]...)
p1yy = diffpoly2(p1, variable_1 = 2, variable_2 = 2); evalpoly(p1yy, [1,1]...)

p1cul = NVarPolynomial{Float64,2}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] )
grad(p1cul, [1,1])
hess(p1cul, [1,1])



# ===================================================================
#  INTEGRATION TEST
# ===================================================================

p1 = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] )
intpoly(p1, lims = [(0,0), (10,2)])

@code_warntype intpoly(p1, lims = [(0,0), (10,2)])


#! INTERPOLATE 3


function datan()
    x = collect(Float64, 1:4);
    y = collect(Float64, 1:4);
    z = collect(Float64, -1:3);
    t = collect(Float64, 0:3);

    m = length(x); n = length(y); p = length(z); q = length(t);
    data = Array{Float64}(undef, m * n * p * q, 5);
    dx  = 2.5;
    dy = -0.4;
    c = 1; 
    for i in 1:m
        for j in 1:n
            for k in 1:p
                for l in 1:q
                    data[c,:] = [x[i], y[j], z[k], t[l], round(x[i]^2 + x[i], digits=2)]';
                    # data[c,:] = [x[i], y[j], z[k], t[l], round(sin(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
                    c += 1;
                end
            end
        end 
    end
    return data
end

function datan_diffx()
    x = collect(Float64, 1:4);
    y = collect(Float64, 1:4);
    z = collect(Float64, -1:3);
    t = collect(Float64, 0:3);

    m = length(x); n = length(y); p = length(z); q = length(t);
    data = Array{Float64}(undef, m * n * p * q, 5);
    dx  = 2.5;
    dy = -0.4;
    c = 1; 
    for i in 1:m
        for j in 1:n
            for k in 1:p
                for l in 1:q
                    data[c,:] = [x[i], y[j], z[k], t[l], round(2*x[i] + 1, digits=2)]';
                    # data[c,:] = [x[i], y[j], z[k], t[l], round(cos(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
                    c += 1;
                end
            end
        end 
    end
    return data
end

function datan_diff2x()
    x = collect(Float64, 1:4);
    y = collect(Float64, 1:4);
    z = collect(Float64, -1:3);
    t = collect(Float64, 0:3);

    m = length(x); n = length(y); p = length(z); q = length(t);
    data = Array{Float64}(undef, m * n * p * q, 5);
    dx  = 2.5;
    dy = -0.4;
    c = 1; 
    for i in 1:m
        for j in 1:n
            for k in 1:p
                for l in 1:q
                    data[c,:] = [x[i], y[j], z[k], t[l], round(2, digits=2)]';
                    # data[c,:] = [x[i], y[j], z[k], t[l], round(cos(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
                    c += 1;
                end
            end
        end 
    end
    return data
end


data = datan()
data = sortslices(data, dims=1);

p1 = interpolaten(data)

grad(p1, [1,1,1,1])
p1x = diffpoly(p1, variable = 1); evalpoly(p1x, [1,1,1,1]...)
p1y = diffpoly(p1, variable = 2); evalpoly(p1y, [1,1,1,1]...)
p1z = diffpoly(p1, variable = 3); evalpoly(p1z, [1,1,1,1]...)
p1t = diffpoly(p1, variable = 4); evalpoly(p1t, [1,1,1,1]...)

hess(p1, [1,1,1,1])
p1xx = diffpoly2(p1, variable_1 = 1, variable_2 = 1); evalpoly(p1xx, [1,1,1,1]...)
p1xy = diffpoly2(p1, variable_1 = 1, variable_2 = 2); evalpoly(p1xy, [1,1,1,1]...)
p1xz = diffpoly2(p1, variable_1 = 1, variable_2 = 3); evalpoly(p1xz, [1,1,1,1]...)
p1xt = diffpoly2(p1, variable_1 = 1, variable_2 = 4); evalpoly(p1xt, [1,1,1,1]...)
p1yx = diffpoly2(p1, variable_1 = 2, variable_2 = 1); evalpoly(p1yx, [1,1,1,1]...)
p1yy = diffpoly2(p1, variable_1 = 2, variable_2 = 2); evalpoly(p1yy, [1,1,1,1]...)
p1yz = diffpoly2(p1, variable_1 = 2, variable_2 = 3); evalpoly(p1yz, [1,1,1,1]...)
p1yt = diffpoly2(p1, variable_1 = 2, variable_2 = 4); evalpoly(p1yt, [1,1,1,1]...)
p1zx = diffpoly2(p1, variable_1 = 3, variable_2 = 1); evalpoly(p1zx, [1,1,1,1]...)
p1zy = diffpoly2(p1, variable_1 = 3, variable_2 = 2); evalpoly(p1zy, [1,1,1,1]...)
p1zz = diffpoly2(p1, variable_1 = 3, variable_2 = 3); evalpoly(p1zz, [1,1,1,1]...)
p1zt = diffpoly2(p1, variable_1 = 3, variable_2 = 4); evalpoly(p1zt, [1,1,1,1]...)
p1tx = diffpoly2(p1, variable_1 = 4, variable_2 = 1); evalpoly(p1tx, [1,1,1,1]...)
p1ty = diffpoly2(p1, variable_1 = 4, variable_2 = 2); evalpoly(p1ty, [1,1,1,1]...)
p1tz = diffpoly2(p1, variable_1 = 4, variable_2 = 3); evalpoly(p1tz, [1,1,1,1]...)
p1tt = diffpoly2(p1, variable_1 = 4, variable_2 = 4); evalpoly(p1tt, [1,1,1,1]...)


data_diffx = datan_diffx()
data_diff2x = datan_diff2x()


approxog = [ evalpoly(p1,x...) for x in zip(data[:,1],data[:,2],data[:,3],data[:,4]) ]
sum( abs.(approxog - data[:,5]) ) / length(data[:,5])


approxog = [ evalpoly(p,x...) for x in zip(data_diffx[:,1],data_diffx[:,2],data_diffx[:,3],data_diffx[:,4]) ]
sum( abs.(approxog - data_diffx[:,5]) ) / length(data_diffx[:,5])

approxog = [ evalpoly(pp,x...) for x in zip(data_diff2x[:,1],data_diff2x[:,2],data_diff2x[:,3],data_diff2x[:,4]) ]
sum( abs.(approxog - data_diff2x[:,5]) ) / length(data_diff2x[:,5])

@benchmark interpolaten(data)
@code_warntype interpolaten(data)

evalpoly(p1, 4,4,1,0)
# 3.36


p1