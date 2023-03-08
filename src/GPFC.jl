
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

include("Posterior.jl")
export kernelfunctionasdd
export Marginal
export Coveriance_energy
export Coveriance_force
export Coveriance_fc2
export Coveriance_fc3
export Posterior

end

