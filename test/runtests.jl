using GPFC
using Test

using KernelFunctions
σₒ = 1.0
l = 0.4
k = σₒ * SqExponentialKernel() ∘ ScaleTransform(l)



@testset "GPFC.jl" begin
    

end
