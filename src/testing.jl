
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

p1 = OneVarPolynomial([0,1,2], [1,3,2] )
p = diffpoly(p1)
p = diffpoly(p)

evalpoly(p, 1)

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

data = data3();
# @benchmark sortslices(data, dims = 1);
data = sortslices(data, dims = 1);
p1 = interpolate2(data)

approxog = [ evalpoly(p1,p...) for p in zip(data[:,1],data[:,2]) ]
sum( abs.(approxog - data[:,3]) ) / length(data[:,3])


@time interpolate2(data); 


# ===================================================================
#  DIFFERENTIATION TEST
# ===================================================================

p1 = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] )
p = diffpoly(p1, variable = 2)
@code_warntype diffpoly(p1, variable = 2)

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
                    data[c,:] = [x[i], y[j], z[k], t[l], round(sin(x[i]) * cos(y[j]) * exp(z[k]) * 5 * (sin(t[l] * pi / 3) + 0.5), digits=2)]';
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


@benchmark interpolaten(data)
@code_warntype interpolaten(data)

evalpoly(p1, 4,4,1,0)
# 3.36