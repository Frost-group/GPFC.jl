
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

using ProgressMeter

include("csv.jl")
export ASEFeatureTarget

include("Posterior.jl")
export kernelfunction1
export kernelfunction2
export kernelfunction3
export kernelfunction4
export Marginal
export Coveriance_energy
export Coveriance_force
export Coveriance_fc2
export Coveriance_fc3
export Posterior_energy
export Posterior_force
export PosteriorFC2
export PosteriorFC3


end

