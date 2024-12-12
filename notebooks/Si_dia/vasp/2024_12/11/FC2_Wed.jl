begin
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum, Statistics
	using CSV, DataFrames, DelimitedFiles
	using Plots, StatsBase
	using ProgressMeter
end

begin
	σₒ = 0.05                  # Kernel Scale
	l = 0.4		    
	Num = 499            # Number of training points
	DIM = 3                     # Dimension of Materials
	model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
	order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
		
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
end;

begin
	σₑ = 1e-5				      # Energy Gaussian noise
	σₙ = 1e-5                   # Force Gaussian noise for Model 2 (σₑ independent)
end

begin
	function kernelfunction1(kernel, x₁, x₂)
		return Zygote.gradient( a -> kernel(a, x₂), x₁)[1]
	end
	function kernelfunction2(kernel, x₁, x₂)
		return Zygote.hessian(a -> kernel(a, x₂), x₁)
	end
	function kernelfunction3(kernel, x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction2(kernel, a, x₂), x₁)
	end
	function kernelfunction4(kernel, x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction3(kernel, a, x₂), x₁)
	end
end

function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, numt::Int64, dimA::Int64)
	a  = 4 - dimA
	feature = (CSV.File(FileFeature)|> Tables.matrix)[begin:a:end,2:numt+1]
	
	equi = feature[:,1]
	
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (CSV.File(FileEnergy)|> Tables.matrix)[begin:numt,2]

	force = -reshape((CSV.File(FileForce)|> Tables.matrix)[begin:a:end,2:numt+1], (dim*num,1))
    #force[1:dim] = zeros(dim)
	
	Target = vcat(energy, reshape(force, (dim*num,1)))
	
	
	return equi, feature, energy, force, Target
end

equi, feature, energy, force, Target = ASEFeatureTarget(
    "feature_vasp", "energy_vasp", "force_vasp", Num, DIM);


function Marginal(kernel, X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
		dim = size(X,1)
		num = size(X,2)
		#building Marginal Likelihood containers
		#For Energy + Force
		KK = zeros(((1+dim)*num, (1+dim)*num))
		#For Energy
		K₀₀ = zeros(((1)*num, (1)*num))
		#For Force
		K₁₁ = zeros(((dim)*num, (dim)*num))
		
		@showprogress "Processing items..." for i in 1:num 
			for j in 1:num 
			#Fillin convarian of Energy vs Energy
				KK[i, j] = kernel(X[:,i], X[:,j])
			#Fillin convarian of Force vs Energy
				KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernelfunction1(kernel, X[:,i], X[:,j])
			#Fillin convarian of Energy vs Force	
				KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j]
			#Fillin convarian of Force vs Force
				KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = -kernelfunction2(kernel, X[:,i], X[:,j])
			end
		end
	
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))
	
		Kₘₘ = KK + II

		println("Marginal Likelihood calculated successfully!")
		return Kₘₘ
end

function Coveriance_fc2(kernel, X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for FC2 prediction
	#building Covariance matrix containers	
	K₂ₙₘ= zeros((dim, dim, (1+dim)*num))
		
	@showprogress "Processing items..." for j in 1:num
		#Fillin convarian of Energy vs FC2
		K₂ₙₘ[:,:,j] = reshape(
					 kernelfunction2(kernel, X[:,j], xₒ)
					, (dim, dim)
				)
		#Fillin convarian of Force vs FC2
		K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					 kernelfunction3(kernel, X[:,j], xₒ)
					, (dim, dim, dim)
				)
	end
	println("Covariance Likelihood calculated successfully!")
	return K₂ₙₘ
end

function PosteriorFC2(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)  
	Kₙₘ = Covariance
	
	MarginalTar = zeros(dimₜ)
	@einsum MarginalTar[m] = Kₘₘ⁻¹[m, n] * Target[n]
	
	size(Kₙₘ) == (dimₚ, dimₚ, dimₜ)
	Meanₚ = zeros(dimₚ, dimₚ)
	@einsum Meanₚ[i, j] = Kₙₘ[i, j, m] * MarginalTar[m]

	println("FC2 calculated successfully!")
	return Meanₚ 
end

function run_with_timer(task_function, args...)
    println("Starting task...")
    elapsed_time = @elapsed results = task_function(args...)
    println("Calculation time: $(elapsed_time) seconds")
	return results 
end

Kₘₘ = run_with_timer(Marginal, kernel, feature, σₑ, σₙ);
K₂ₙₘ = run_with_timer(Coveriance_fc2, kernel, feature, equi);
FC2 =run_with_timer(PosteriorFC2, Kₘₘ, K₂ₙₘ, Target);

FC2