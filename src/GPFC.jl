
module GPFC


using KernelFunctions
using ForwardDiff
using Plots
using LinearAlgebra
using Einsum
using CSV
using DataFrames
using DelimitedFiles
using Optim
using HDF5


include("FeatureTarget.jl")
export ASEFeatureTarget

include("DerivativeKernel.jl")
export kernel

include("Marginal.jl") 
export invMarginal

include("Covariant.jl")
export Covariant


include("Posterior.jl")
export Posterior

end