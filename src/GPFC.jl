module GPFC

using LinearAlgebra
using CSV, Tables
using KernelFunctions
using ForwardDiff, Zygote
using Einsum
using ProgressMeter

# Reading JuLIP/extxyz datasets needs JuLIP; the method lives in ext/GPFCJuLIPExt.jl
# and loads automatically after `using JuLIP`.
function Read_JuLIP_Atoms end
export Read_JuLIP_Atoms

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

include("Analy_kernel.jl")
export rbf_kernel
export kernel_1st_derivative
export kernel_2nd_derivative
export kernel_3rd_derivative
export kernel_4th_derivative

end

