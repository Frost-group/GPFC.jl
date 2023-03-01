
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

#include("DerivativeKernel.jl")
#export kernel

include("Posterior.jl")
export f1st
export f2nd
export f3rd
export f4th
export Marginal
export Covariant
export PosteriorMean

include("Posterior2.jl")
export kernelfunction
export Marginal2
export Covariant2
export PosteriorMean2
end

