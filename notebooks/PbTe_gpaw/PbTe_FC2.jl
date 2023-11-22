### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 346974a0-8249-11ee-075d-33c092b30816
begin
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum
	using CSV
	using DataFrames
	using DelimitedFiles
	using Plots
end

# ╔═╡ e79b6a06-5285-44b2-81ea-e3a96247db8f
function kernelfunction(k, x₁, x₂, grad::Int64)
	function f1st(x₁, x₂) 
		Zygote.gradient( a -> k(a, x₂), x₁)[1]
	end	
	function f2nd(x₁, x₂)
		Zygote.hessian(a -> k(a, x₂), x₁)
	end
	function f3rd(x₁, x₂) 
		ForwardDiff.jacobian( a -> f2nd(a, x₂), x₁)
	end 
	function f4th(x₁, x₂)
		ForwardDiff.jacobian( a -> f3rd(a, x₂), x₁)
	end

	if grad == 0
		return k(x₁, x₂)
	elseif grad == 1
		return f1st(x₁, x₂)
	elseif grad == 2
		return f2nd(x₁, x₂)	
	elseif grad == 3
		return f3rd(x₁, x₂)
	elseif grad == 4
		return f4th(x₁, x₂)
	else
		println("Grad in btw [0, 4]")
	end
end

# ╔═╡ e1d8fc9e-00fd-4fd9-975d-6b6a4f79068a
function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, numt::Int64, dimA::Int64)
	a  = 4 - dimA
	feature = (CSV.File(FileFeature)|> Tables.matrix)[begin:a:end,2:numt+1]
	
	equi = feature[:,1]
	
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (CSV.File(FileEnergy)|> Tables.matrix)[begin:numt,2]

	force = -reshape((CSV.File(FileForce)|> Tables.matrix)[begin:a:end,2:numt+1], (dim*num,1))
		
	Target = vcat(energy, reshape(force, (dim*num,1)))
	
	
	return equi, feature, energy, force, Target
end

# ╔═╡ cbd94f02-b08d-4208-9e2a-b24a35d2646a
function Marginal(X::Matrix{Float64}, k, l::Float64, σₑ::Float64, σₙ::Float64)
	dim = size(X,1)
	num = size(X,2)
	#building Marginal Likelihood containers
	#For Energy + Force
	KK = zeros(((1+dim)*num, (1+dim)*num))
	#For Energy
	K₀₀ = zeros(((1)*num, (1)*num))
	#For Force
	K₁₁ = zeros(((dim)*num, (dim)*num))
	
	for i in 1:num 
		for j in 1:num 
		#Fillin convarian of Energy vs Energy
			KK[i, j] = kernelfunction(k, X[:,i], X[:,j], 0)
		#Fillin convarian of Force vs Energy
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernelfunction(k, X[:,i], X[:,j], 1)
		#Fillin convarian of Energy vs Force	
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j]
		#Fillin convarian of Energy vs Force
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = -kernelfunction(k, X[:,i], X[:,j], 2)
		end
	end

	Iee = σₑ^2 * Matrix(I, num, num)
	Iff = (σₑ / l)^2 * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

	Kₘₘ = KK + II
	
	return Kₘₘ
end

# ╔═╡ e9efaf81-6634-435c-8030-45cc949d8068
function Coveriance_fc2(X::Matrix{Float64}, xₒ::Vector{Float64}, k)
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for FC2 prediction
	#building Covariance matrix containers	
	K₂ₙₘ= zeros((dim, dim, (1+dim)*num))
		
	for j in 1:num
		#Fillin convarian of Energy vs FC2
		K₂ₙₘ[:,:,j] = reshape(
					 kernelfunction(k, X[:,j], xₒ, 2)
					, (dim, dim)
				)
		#Fillin convarian of Force vs FC2
		K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					 kernelfunction(k, X[:,j], xₒ, 3)
					, (dim, dim, dim)
				)
	end
	return K₂ₙₘ
end

# ╔═╡ eb89d8e7-8650-4f6d-8894-2c7ad658f616
begin
	σₒ = 0.05                   # Kernel Scale
	l = 0.43                	# Length Scale
	σₑ = 1e-5 					# Energy Gaussian noise
	σₙ = 1e-6                   # Force Gaussian noise for Model 2 (σₑ independent)
		
	Num = 199                 # Number of training points
	DIM = 3                     # Dimension of Materials
	model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
	order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
		
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
end;

# ╔═╡ 79d1e9f9-56d1-4425-9362-ff86a5c452f5
equi, feature, energy, force, Target = ASEFeatureTarget(
    "feature", "energy", "force", Num, DIM);

# ╔═╡ 25f5cb64-40e5-4a2a-9a0d-42878a3aa9ae
@time Kₘₘ = Marginal(feature, kernel, l, σₑ, σₙ);

# ╔═╡ b2ced1ff-11f5-46df-9e4a-a961ab8f0a63
@time K₂ₙₘ = Coveriance_fc2(feature, equi, kernel);

# ╔═╡ b969f033-376e-4db8-a955-9dcab0c79378
function Posterior(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)   #
	Kₙₘ = Covariance
	
	MarginalTar = zeros(dimₜ)
	@einsum MarginalTar[m] = Kₘₘ⁻¹[m, n] * Target[n]

	if size(Kₙₘ) == (dimₜ,)
		Meanₚ = Kₙₘ'  * MarginalTar
		
	elseif size(Kₙₘ) == (dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ)
		@einsum Meanₚ[i] = Kₙₘ[i, m] * MarginalTar[m]
	
	elseif size(Kₙₘ) == (dimₚ, dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ, dimₚ)
		@einsum Meanₚ[i, j] = Kₙₘ[i, j, m] * MarginalTar[m]

	elseif size(Kₙₘ) == (dimₚ, dimₚ, dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ, dimₚ, dimₚ)
		@einsum Meanₚ[i, j, k] = Kₙₘ[i, j, k, m] * MarginalTar[m]
	end

	return Meanₚ 
end

# ╔═╡ be8cba06-01f0-4ad5-861f-5a4c556a9cc4
FC2= Posterior(Kₘₘ, K₂ₙₘ, Target);

# ╔═╡ 4451bffe-18c9-4da3-ba22-9df2cb2e6472
heatmap(1:size(FC2[:,:],1),
	    1:size(FC2[:,:],2), FC2[:,:],
	    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
	    xlabel="feature coord. (n x d)",
		ylabel="feature coord. (n x d)",
		aspectratio=:equal,
		size=(700, 700),
	    title="PbTe_FC2 (Traning Data = " *string(199) *")" )

# ╔═╡ Cell order:
# ╠═346974a0-8249-11ee-075d-33c092b30816
# ╠═e79b6a06-5285-44b2-81ea-e3a96247db8f
# ╠═e1d8fc9e-00fd-4fd9-975d-6b6a4f79068a
# ╠═cbd94f02-b08d-4208-9e2a-b24a35d2646a
# ╠═e9efaf81-6634-435c-8030-45cc949d8068
# ╠═eb89d8e7-8650-4f6d-8894-2c7ad658f616
# ╠═79d1e9f9-56d1-4425-9362-ff86a5c452f5
# ╠═25f5cb64-40e5-4a2a-9a0d-42878a3aa9ae
# ╠═b2ced1ff-11f5-46df-9e4a-a961ab8f0a63
# ╠═b969f033-376e-4db8-a955-9dcab0c79378
# ╠═be8cba06-01f0-4ad5-861f-5a4c556a9cc4
# ╠═4451bffe-18c9-4da3-ba22-9df2cb2e6472
