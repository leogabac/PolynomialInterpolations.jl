x = 1:3 |> collect
# 0.0
# 1.0
# 2.0
# 3.0
# 4.0
# 5.0
# 6.0
# 7.0
# 8.0
# 9.0
# 10.0

composite = [ x.^k for k in 0:(length(x)-1)  ]
hcat(composite...)
# ::Matrix{eltype(eltype(composite))}


test_poly = OneVarPolynomial{Float64}([1, 4], [1., 2.5])

test_poly.coefficients
test_poly.pows

x = 1:3 |> collect
y = x[1:end-1]
orders = (length(x)-1, length(y)-1 )
ranges = ( 0:m for m in reverse(orders) ) |> collect
powers = vec( reverse.( collect( Iterators.product(ranges...) ) ) )
coef = [3,1,4,1,5,9.]
test_poly_2d = TwoVarPolynomial{eltype(coef)}(powers, coef)

x_test = [2., 3]
x_test::Real...
pure = [ *( (x_test.^currentpow)...) for currentpow in powers  ]
sum( test_poly_2d.coefficients .* pure  )


function test_evalpoly(p::TwoVarPolynomial,x::Real...)
    # pure = vec( [x^j * y^k for k in 0:p.order[1], j in 0:p.order[2] ] ) # pure powers, no coefficients included
    pure = [ *( (x.^currentpow)...) for currentpow in p.pows  ]
    return sum( p.coefficients .* pure  )
end

p1
test_poly_2d
test_evalpoly(test_poly_2d, x_test...)
coeffs = Ainv*data[:,end]
NVarPolynomial{eltype(coeffs),4}( powers , coeffs )

# symmetric matrix
test = [(i<=j) ? rand() : nothing for i ∈ 1:3, j ∈ 1:3]
test = [(!isnothing(test[i,j])) ? test[i,j] : test[j,i] for i in eachindex(eachrow(test)), j in eachindex(eachcol(test))]

eachindex(eachrow(test))
eachindex(eachcol(test))

poly
nvars = 4
# calculo df/dx_i para cada variable y voy guardando en un vector
gradient = [diffpoly(poly, variable = i) for i in 1:nvars]::Vector{typeof(poly)}
gradient isa Vector{NVarPolynomial{Float64,4}}



#= ==========================================================================================
=============================================================================================
1D
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos

p1 = OneVarPolynomial([0,1,2], [1,3,2] ) # f(x) = 1 + 3x + 2x²
grad(p1, [1]) # f'(x) = 3 + 4x => f'(1) = 7
gradient = grad(p1)
evalgrad(gradient,[1])

hess(p1, [1])
hessian = hess(p1)
evalhess(hessian, [1])

#= ==========================================================================================
=============================================================================================
2D
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos, además comparo con el método numérico explícito y hecho paso a paso

p2 = TwoVarPolynomial{Float64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy

grad(p2, [1,1]) # ∇f = [3 + 4y, 2 + 4x] => ∇f([1,1]) = [7, 6]
gradient = grad(p2) # ∇f = [3 + 4y, 2 + 4x] => ∇f([1,1]) = [7, 6]
evalgrad(gradient, [1,1])

hess(p2, [1,1]) 
hessian = hess(p2)
evalhess(hessian, [1,1])

#= ==========================================================================================
=============================================================================================
ND - 1
=============================================================================================
========================================================================================== =#

# meto un polinomio a mano y comparo resultados numéricos con los analíticos, además comparo con el método numérico explícito y hecho paso a paso

pn_1 = NVarPolynomial{Float64,2}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy
grad(pn_1, [1,1])
gradient = grad(pn_1) 
evalgrad(gradient, [1,1])

hess(pn_1, [1,1])
hessian = hess(pn_1)
evalhess(hessian, [1,1])



p1 = OneVarPolynomial([0,1,2], [1,3,2] ) # f(x) = 1 + 3x + 2x²
p2 = OneVarPolynomial([0,1,4], [1,5,7] ) # g(x) = 1 + 5x + 7x⁴

x = rand()
evalpoly(p1, x) + evalpoly(p2, x)
evalpoly(p1 + p2, x)
evalpoly(p1, x) - evalpoly(p2, x)
evalpoly(p1 - p2, x)
p1 - p1

grad(p1)
hess(p1)
laplacian(p1)

p1 = TwoVarPolynomial{Int64}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy
p2 = TwoVarPolynomial{Int64}([(0,0), (0,1), (1,0), (2,1)], [1,3,5,16] ) # 1 + 3y + 5x + 16x²y

x = rand(2)
evalpoly(p1, x...) + evalpoly(p2, x...)
evalpoly(p1 + p2, x...)
evalpoly(p1, x...) - evalpoly(p2, x...)
evalpoly(p1 - p2, x...)
p1 - p1

diffpoly2(p2, variable_1 = 1, variable_2 = 1)
diffpoly2(p2, variable_1 = 2, variable_2 = 2)
laplacian(p2)


p1 = NVarPolynomial{Int64,2}([(0,0), (0,1), (1,0), (1,1)], [1,2,3,4] ) # 1 + 2y + 3x + 4xy
p2 = NVarPolynomial{Int64,2}([(0,0), (0,1), (1,0), (2,1)], [1,3,5,16] ) # 1 + 3y + 5x + 16x²y

x = rand(2)
evalpoly(p1, x...) + evalpoly(p2, x...)
evalpoly(p1 + p2, x...)
evalpoly(p1, x...) - evalpoly(p2, x...)
evalpoly(p1 - p2, x...)
p1 - p1


# tuple_length(::NTuple{N, Any}) where {N} = Val{N}()

# function tuple_type_length(x) 
# x = hessian[1].pows
# eltype(x)
#     n = -1; 
#     while !(eltype(x) <: NTuple{n+=1, Any}) 
#     end 
#     return n
# # end
hessian[1]
pows = hessian[1].pows;
fieldcount(eltype(pows))

fieldcount(typeof(hessian[1]))


p = hessian[1]

isempty(p.pows)

[zeros(Int64, fieldcount(eltype(p.pows))) |> Tuple]

p = hessian[1]

new_pows = p.pows; new_coef = p.coefficients;

if isempty(new_pows)
    new_pows = [zeros(Int64, fieldcount(eltype(new_pows))) |> Tuple]
    new_coef = [0] .|> eltype(p.coefficients)
end
new_pows
new_coef
NVarPolynomial{eltype(new_coef), length(new_pows[1])}(new_pows, new_coef)



hessian[1] |> empty2zero_poly
hessian[2] |> empty2zero_poly
p = hessian[2]

[0] .|> eltype(p.coefficients)

struct TwoVarPolynomial{T}
    pows::Vector{Tuple{Int64,Int64}}
    coefficients::Vector{T}
end

f = NVarPolynomial{Float64,2}([(0,0), (0,1), (1,0), (1,1), (3,0)], [1,2,3,4,2] ) # 1 + 2y + 3x + 4xy + + 2x³
f |> typeof
grad_f = grad(f)|> typeof # ∇f = [3 + 4y + 6x², 2 + 4x]
hess_f = hess(f) |> typeof# hess(f) = [12x, 4; 4, 0]
lap_f = laplacian(f)|> typeof # ∂²/∂x² f + ∂²/∂y² f = 12x




OneVarPoly_n_Diff{Int64}(f, grad_f, hess_f, lap_f)