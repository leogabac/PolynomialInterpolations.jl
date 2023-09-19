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