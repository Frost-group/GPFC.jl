using GPFC
using Test

#Setting Kernel function and its derivatives with automaic differentiation
using KernelFunctions, ForwardDiff, Zygote
#Doing linear algebra and tensor contraction
using LinearAlgebra, Einsum
#Pulling Dataset from CSV
using CSV, DataFrames, DelimitedFiles

σₒ = 0.1                   # Kernel Scale
l = 0.4                     # Length Scale
σₑ = 1e-5                   # Energy Gaussian noise
σₙ = 1e-6                   # Force Gaussian noise for Model 2 (σₑ independent)
		
Num = 1                   # Number of training points
DIM = 3                     # Dimension of Materials
#model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
#order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
	
#Defining Kernel function
kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

#set file name
Featurefile = "feature_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
Energyfile = "energy_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
Forcefile = "force_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"

#Reading file from CSV
equi, feature, energy, force, Target = ASEFeatureTarget(Featurefile, Energyfile, Forcefile, Num, DIM)
#Obtain DOF of the problem
diml = size(feature, 1)

#Testing posterior FC2 size and Time for its Calculation! 
@time Kmm = Marginal(feature, kernel, l, σₑ);
@time K2nm = Coveriance_fc2(feature, equi, kernel);
@time pMean_fc2 = Posterior(Kmm, K2nm, Target)

@testset "FC2" begin
    @test size(Kmm) == ( (diml+1)*Num, (diml+1)*Num )
    @test size(K2nm) == ( diml, diml, (diml+1)*Num)
    @test size(pMean_fc2) == (diml, diml)
end


