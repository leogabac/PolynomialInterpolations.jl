
# ===================================================================
#  STRUCTS
# ===================================================================

# Polynomial data structs

struct OneVarPolydata{T}
    x::Vector{T}
    y::Vector{T}
end

struct TwoVarPolydata{T}
    x1::Vector{T}
    x2::Vector{T}
    y::Vector{T}
end

struct NVarPolydata{T,N}
    x::Vector{Vector{T}}
    y::Vector{T}
end

# Polynomial "object" structs

struct OneVarPolynomial{T}
    pows::Vector{Int64}
    coefficients::Vector{T}
end

struct TwoVarPolynomial{T}
    pows::Vector{Tuple{Int64,Int64}}
    coefficients::Vector{T}
end

struct NVarPolynomial{T,N}
    pows::Vector{Tuple{Vararg{Int64,N}}}
    coefficients::Vector{T}
end

# Polynomial "object" structs with results from differential analysis

struct OneVarPoly_n_Diff{T}
    poly::OneVarPolynomial{T}
    grad::Vector{OneVarPolynomial{T}}
    hess::Matrix{OneVarPolynomial{T}}
    lap::OneVarPolynomial{T}
end

struct TwoVarPoly_n_Diff{T}
    poly::TwoVarPolynomial{T}
    grad::Vector{TwoVarPolynomial{T}}
    hess::Matrix{TwoVarPolynomial{T}}
    lap::TwoVarPolynomial{T}
end

struct NVarPoly_n_Diff{T,N}
    poly::NVarPolynomial{T,N}
    grad::Vector{NVarPolynomial{T,N}}
    hess::Matrix{NVarPolynomial{T,N}}
    lap::NVarPolynomial{T,N}
end

# ===================================================================
#  CONSTRUCTORS
# ===================================================================







# ===================================================================
#  "ACCESS THE FIELD" RELATED FUNCTIONS
# ===================================================================
"""
    independent(data)

Retrieves the unique data from the independent variable of a `Polydata` type.

# Examples
```julia-repl
julia> data = [0 0 5; 0 1 6; 1 0 7; 1 1 0];
4×3 Matrix{Int64}:
 0  0  5
 0  1  6
 1  0  7
 1  1  8

julia> d = dataloader(data);

julia> typeof(d)
TwoVarPolydata{Int64}

julia> independent(d);
([0, 1], [0, 1])
```
"""
independent(data::OneVarPolydata) = data.x
independent(data::TwoVarPolydata) = (data.x1, data.x2)
independent(data::NVarPolydata) = Tuple(data.x)


"""
    dependent(data)

Retrieves the unique data from the dependent variable of a `Polydata` type.

# Examples
```julia-repl
julia> data = [0 0 5; 0 1 6; 1 0 7; 1 1 0];
4×3 Matrix{Int64}:
 0  0  5
 0  1  6
 1  0  7
 1  1  8

julia> d = dataloader(data);

julia> typeof(d)
TwoVarPolydata{Int64}

julia> dependent(d);
[5, 6, 7, 8]
```
"""
dependent(data::OneVarPolydata) = data.y
dependent(data::TwoVarPolydata) = data.y
dependent(data::NVarPolydata) = data.y

