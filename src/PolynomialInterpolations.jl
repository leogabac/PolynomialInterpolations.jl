module PolynomialInterpolations

    using Base: Tuple
    import Base: Iterators.take, Iterators.rest, Base.+, Base.-
    using LinearAlgebra: eltype
    using Kronecker, LinearAlgebra, FiscomTools;

    include("structs.jl")
    include("dataloaders.jl")
    include("interpolate.jl")
    include("algebra.jl")
    include("differential operators.jl")


    # Variable types
    export OneVarPolydata, TwoVarPolydata, NVarPolydata
    export OneVarPolynomial, TwoVarPolynomial, NVarPolynomial
    export OneVarPoly_n_Diff, TwoVarPoly_n_Diff, NVarPoly_n_Diff
    # Interpolation
    export interpolate
    # Differentiatial analysis
    export diffpoly, gradient, hessian, laplacian, diff_analyze
    # Evaluation functions
    export evalgrad, evalhess #Base.evalpoly, Base.+, Base.- se exportan solas?
    # Data Loaders
    export dataloader
    export independent, dependent

end