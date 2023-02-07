using GPFC
using Test

using KernelFunctions
using ForwardDiff
using LinearAlgebra
σₒ = 1.0
l = 0.4
k = σₒ * SqExponentialKernel() ∘ ScaleTransform(l)

#@test 

#@testset "GPFC.jl" begin
    

#end
