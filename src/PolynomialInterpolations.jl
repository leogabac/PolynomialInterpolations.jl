module PolynomialInterpolations

    import Base: Tuple
    import Base: Iterators.take, Iterators.rest
    import LinearAlgebra: eltype
    import Kronecker
    import LinearAlgebra
    
    include("structs.jl")
    include("dataloaders.jl")
    include("interpolate1.jl")
    include("interpolate2.jl")
    include("interpolaten.jl")

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
