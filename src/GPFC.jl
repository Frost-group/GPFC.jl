
module GPFC

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

using Distributed
# Writing FC2 + FC3 for phononpy
using HDF5


include("ASEreading.jl")
export ASEFeatureTarget

#include("Posterior.jl")
#export f1st
#export f2nd
#export f3rd
#export f4th
#export Marginal
#export Covariant
#export PosteriorMean

#include("Posterior2.jl")
#export kernelfunction
#export Marginal2
#export Covariant2
#export PosteriorMean2

include("Posterior3.jl")
export kernelfunctionasdd
export Marginal
export Coveriance_energy
export Coveriance_force
export Coveriance_fc2
export Coveriance_fc3
export Posterior

end

