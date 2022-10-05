using GPFC
using Test

σₒ = 1.0
l = 0.4
k = σₒ * SqExponentialKernel() ∘ ScaleTransform(l)

@test 

@testset "GPFC.jl" begin
    

end
