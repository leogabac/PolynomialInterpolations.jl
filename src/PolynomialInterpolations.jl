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
    # Interpolation
    export interpolate1, interpolate2, interpolaten
    # Differentiation, integration
    export diffpoly, intpoly
    # Data Loaders, DO NOT WORK
    export dataloader
    export independent, dependent

end # module PolynomialInterpolations
