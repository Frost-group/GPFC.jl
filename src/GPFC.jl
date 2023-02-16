
module GPFC


using KernelFunctions
using ForwardDiff
using LinearAlgebra
using Einsum
using CSV
using DataFrames
using Optim
using Distributed
using HDF5


include("ASEreading.jl")
export ASEFeatureTarget

include("DerivativeKernel.jl")
export kernel

include("Posterior.jl")
export MarginalLike
export CovariantMatrix
export PosteriorMean

end

