
module GPFC


using KernelFunctions
using ForwardDiff
using Plots
using LinearAlgebra
using YAML
using HDF5

include("ReadnWrite.jl")


include("DerivativeKernel.jl")
export kernel

include("Marginal.jl") 
export Marginal

include("Covariant.jl")
export Covariant


include("Posterior.jl")
export Posterior

end