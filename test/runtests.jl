using GPFC
using Test
using LinearAlgebra, KernelFunctions

println("GPFC tests, against the bulk-silicon dataset in test/.")

datafile(name) = joinpath(@__DIR__, name * "_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv")

@testset verbose = true "GPFC" begin

    # Analytic RBF derivatives must agree with the automatic-differentiation path.
    # SqExponentialKernel() ∘ ScaleTransform(s) is exp(-s²|x-x'|²/2), so s = 1/l.
    @testset "analytic kernel derivatives = AD" begin
        σₒ, l = 0.7, 0.4
        kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(1 / l)
        x, x′ = [0.10, -0.20, 0.05], [0.00, 0.10, -0.15]
        n = length(x)

        @test rbf_kernel(x, x′, σₒ, l) ≈ kernel(x, x′)
        @test kernel_1st_derivative(x, x′, σₒ, l) ≈ kernelfunction1(kernel, x, x′)
        @test kernel_2nd_derivative(x, x′, σₒ, l) ≈ kernelfunction2(kernel, x, x′)
        @test kernel_3rd_derivative(x, x′, σₒ, l) ≈ reshape(kernelfunction3(kernel, x, x′), n, n, n)
        @test kernel_4th_derivative(x, x′, σₒ, l) ≈ reshape(kernelfunction4(kernel, x, x′), n, n, n, n)
    end

    # Fit the GP on energies and forces of rattled Si supercells.
    σₒ, l = 0.1, 0.4      # kernel scale, length scale
    σₑ, σₙ = 1e-5, 1e-6   # energy and force noise
    kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    num = 5
    equi, feature, energy, force, Target =
        ASEFeatureTarget(datafile("feature"), datafile("energy"), datafile("force"), num, 3)
    dim = size(feature, 1)
    Kmm = Marginal(kernel, feature, σₑ, σₙ)

    @testset "marginal likelihood matrix" begin
        @test size(Kmm) == ((dim + 1) * num, (dim + 1) * num)
        @test Kmm ≈ Kmm'
        @test isposdef(Symmetric(Kmm))
    end

    # The posterior mean must reproduce a training point (noise ≪ kernel scale).
    @testset "posterior interpolates training data" begin
        x₂ = feature[:, 2]
        @test Posterior_energy(Kmm, Coveriance_energy(kernel, feature, x₂), Target) ≈ energy[2] rtol = 1e-6
        ∇E₂ = Target[num+1+dim:num+2dim]
        @test Posterior_force(Kmm, Coveriance_force(kernel, feature, x₂), Target) ≈ ∇E₂ rtol = 1e-3
    end

    @testset "FC2 is a symmetric Hessian" begin
        Φ₂ = PosteriorFC2(Kmm, Coveriance_fc2(kernel, feature, equi), Target)
        @test size(Φ₂) == (dim, dim)
        @test Φ₂ ≈ Φ₂'
    end

    # FC3 needs fourth derivatives of a dim-dimensional kernel, so keep it small.
    @testset "FC3 is symmetric in all indices" begin
        n₃ = 2
        e₃, f₃, _, _, T₃ = ASEFeatureTarget(datafile("feature"), datafile("energy"), datafile("force"), n₃, 3)
        Φ₃ = PosteriorFC3(Marginal(kernel, f₃, σₑ, σₙ), Coveriance_fc3(kernel, f₃, e₃), T₃)
        @test size(Φ₃) == (dim, dim, dim)
        @test Φ₃ ≈ permutedims(Φ₃, (2, 1, 3))
        @test Φ₃ ≈ permutedims(Φ₃, (3, 2, 1))
    end
end
