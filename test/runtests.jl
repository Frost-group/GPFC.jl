using GPFC
using Test

#Setting Kernel function and its derivatives with automaic differentiation
using KernelFunctions, ForwardDiff, Zygote
#Doing linear algebra and tensor contraction
using LinearAlgebra, Einsum
#Pulling Dataset from CSV
using CSV, DataFrames, DelimitedFiles

@testset verbose = true "GPFC" begin
    @testset verbose = true "$i Datapoints" for i in 1:10
    #Defining all parameters    
        σₒ = 0.1                    # Kernel Scale
        l = 0.4                     # Length Scale
        σₑ = 1e-5                   # Energy Gaussian noise
        σₙ = 1e-6                   # Force Gaussian noise for Model 2 (σₑ independent)
		
        Num = i                     # Number of training points
        DIM = 3                     # Dimension of Materials object

    #Defining Kernel function
        kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    #Setting file name
        Featurefile = "feature_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
        Energyfile = "energy_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
        Forcefile = "force_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
    #Reading file from CSV
        equi, feature, energy, force, Target = ASEFeatureTarget(Featurefile, Energyfile, Forcefile, Num, DIM)
    #Obtain DOF of the problem    
        diml = size(feature, 1)

    #Calculate Marginal Likelihood 
        @test size(Marginal(feature, kernel, l, σₑ)) == ( (diml+1)*Num, (diml+1)*Num )
        Kmm = Marginal(feature, kernel, l, σₑ)
    
    #Calculate Covariance for predicting PES     
        @testset "FC0" begin
            @test size(Coveriance_energy(feature, equi, kernel)) == ((diml+1)*Num,)
            K0nm = Coveriance_energy(feature, equi, kernel)
            @test size(K0nm' * inv(Kmm) *Target) == (1,1)
        end

    #Calculate Covariance for predicting Forces 
        @testset "FC1" begin
            @test size(Coveriance_force(feature, equi, kernel)) == ( diml, (diml+1)*Num)
            K1nm = Coveriance_force(feature, equi, kernel)
            @test size(Posterior(Kmm, K1nm, Target)) == (diml,)
        end

    #Calculate Covariance for predicting FC2
        @testset  "FC2" begin
            @test size(Coveriance_fc2(feature, equi, kernel)) == ( diml, diml, (diml+1)*Num)
            K2nm = Coveriance_fc2(feature, equi, kernel)
            @test size(Posterior(Kmm, K2nm, Target)) == (diml, diml)
        end

    #Calculate Covariance for predicting FC3
        @testset "FC3" begin
            @test size(Coveriance_fc3(feature, equi, kernel)) == ( diml, diml, diml, (diml+1)*Num)
            K3nm = Coveriance_fc3(feature, equi, kernel)
            @test size(Posterior(Kmm, K3nm, Target)) == ( diml, diml, diml)
        end
    end 
end;
