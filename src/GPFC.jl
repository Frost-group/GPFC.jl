
module GPFC

using KernelFunctions
using ForwardDiff
using Plots
using LinearAlgebra

export kernel
include("DerivativeKernel")

export Marginal
include("Marginal")

export Covariant
include("Covariant")


export Covariant
include("Covariant")

export Posterior
include("Posterior")

end