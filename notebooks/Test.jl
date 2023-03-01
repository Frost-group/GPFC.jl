using KernelFunctions
using ForwardDiff
using LinearAlgebra
using Einsum
using CSV
using DataFrames
using DelimitedFiles


σₒ = 0.1                    # Kernel Scale
l = 0.4                     # Length Scale
σₑ = 1e-5                   # Energy Gaussian noise
σₙ = 1e-6                   # Force Gaussian noise for Model 2 (σₑ independent)
	
Num = 60                    # Number of training points
DIM = 3                     # Dimension of Materials
model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
	
kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

#set file name
Featurefile = "~/Documents/GitHub/GPFC.jl/Dataset/Si-Bulk/n100/Si_feature_222spc_01_n100_PW800_kpts10_e100_d1.csv"
Energyfile = "~/Documents/GitHub/GPFC.jl/Dataset/Si-Bulk/n100/Si_energy_222spc_01_n100_PW800_kpts9_e100_d1.csv"
Forcefile = "~/Documents/GitHub/GPFC.jl/Dataset/Si-Bulk/n100/Si_forces_222spc_01_n100_PW800_kpts9_e100_d1.csv"

#Reading file from CSV
equi, feature, energy, force, Target = ASEFeatureTarget(
    Featurefile, Energyfile, Forcefile, Num, DIM)

#Calculation   
FC, K₀₀, K₁₁, Kₘₘ, Kₙₘ = PosteriorMean(
    feature, equi, Target,
    l, σₑ, σₙ, order, model)