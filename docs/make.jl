using Documenter
using PolynomialInterpolations

makedocs(
    sitename = "PolynomialInterpolations",
    format = Documenter.HTML(),
    modules = [PolynomialInterpolations]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
