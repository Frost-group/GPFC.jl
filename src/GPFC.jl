
module GPFC

<<<<<<< HEAD
# Kernel Functions
using KernelFunctions
# Automatic differentiation
using ForwardDiff, Zygote
# Linear algebra and Tensor contraction
using LinearAlgebra, Einsum
# Get Dataset
using CSV, DataFrames
# Optimization 
using Optim

=======

using KernelFunctions, ForwardDiff
using Einsum, LinearAlgebra
using CSV
using DataFrames
using Optim
using Zygote
>>>>>>> c8895da47bb2758f8d880527221306c496057723
using Distributed
# Writing FC2 + FC3 for phononpy
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

