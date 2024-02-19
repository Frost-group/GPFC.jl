### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 803d8357-aecc-41b0-ae4d-f2164f0e4e6b
import Pkg; Pkg.add("KernelFunctions")

# ╔═╡ 346974a0-8249-11ee-075d-33c092b30816
begin
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum, Statistics
	using CSV
	using DataFrames
	using DelimitedFiles
	using Plots
end

# ╔═╡ d8b8b25d-e716-47e9-b2e4-12731df9e7c0
begin
	σₒ = 0.05                  # Kernel Scale
	l = 0.2			    # Length Scale
		
	Num = 50                 # Number of training points
	DIM = 3                     # Dimension of Materials
	model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
	order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
		
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
end;

# ╔═╡ 39a72f49-6efb-4ad5-a3b0-7b6a6954d6a6
begin
	σₑ = 1e-5				      # Energy Gaussian noise
	σₙ = 1e-10/l                   # Force Gaussian noise for Model 2 (σₑ independent)
end

# ╔═╡ e79b6a06-5285-44b2-81ea-e3a96247db8f
begin
	function kernelfunction1(x₁, x₂)
		return Zygote.gradient( a -> kernel(a, x₂), x₁)[1]
	end
	function kernelfunction2(x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction1(a, x₂), x₁)
	end
	function kernelfunction3(x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction2(a, x₂), x₁)
	end
	function kernelfunction4(x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction3(a, x₂), x₁)
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
    #force[1:dim] = zeros(dim)
	
	Target = vcat(energy, reshape(force, (dim*num,1)))
	
	
	return equi, feature, energy, force, Target
end

# ╔═╡ a6a54c75-f12d-4694-a876-30f0a14ad8a1
equi, feature, energy, force, Target = ASEFeatureTarget(
    "feature_new", "energy_new", "force_new", Num, DIM);

# ╔═╡ cbd94f02-b08d-4208-9e2a-b24a35d2646a
function Marginal(X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
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
			KK[i, j] = kernel(X[:,i], X[:,j])
		#Fillin convarian of Force vs Energy
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernelfunction1(X[:,i], X[:,j])
		#Fillin convarian of Energy vs Force	
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j]
		#Fillin convarian of Force vs Force
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = -kernelfunction2(X[:,i], X[:,j])
		end
	end

	Iee = σₑ^2 * Matrix(I, num, num)
	Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

	Kₘₘ = KK + II
	
	return Kₘₘ
end

# ╔═╡ e9efaf81-6634-435c-8030-45cc949d8068
function Coveriance_fc2(X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for FC2 prediction
	#building Covariance matrix containers	
	K₂ₙₘ= zeros((dim, dim, (1+dim)*num))
		
	for j in 1:num
		#Fillin convarian of Energy vs FC2
		K₂ₙₘ[:,:,j] = reshape(
					 kernelfunction2(X[:,j], xₒ)
					, (dim, dim)
				)
		#Fillin convarian of Force vs FC2
		K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					 kernelfunction3(X[:,j], xₒ)
					, (dim, dim, dim)
				)
	end
	return K₂ₙₘ
end

# ╔═╡ 79d1e9f9-56d1-4425-9362-ff86a5c452f5
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

# ╔═╡ 25f5cb64-40e5-4a2a-9a0d-42878a3aa9ae
@time Kₘₘ = Marginal(feature, σₑ, σₙ);

# ╔═╡ b2ced1ff-11f5-46df-9e4a-a961ab8f0a63
@time K₂ₙₘ = Coveriance_fc2(feature, equi);

# ╔═╡ be8cba06-01f0-4ad5-861f-5a4c556a9cc4
FC2 = Posterior(Kₘₘ, K₂ₙₘ, Target);

# ╔═╡ 4451bffe-18c9-4da3-ba22-9df2cb2e6472
heatmap(1:size(FC2[:,:],1),
	    1:size(FC2[:,:],2), FC2[:,:],
	    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
	    xlabel="feature coord. (n x d)",
		ylabel="feature coord. (n x d)",
		aspectratio=:equal,
		size=(700, 700),
	    title="PbTe_FC2 (Traning Data = " *string(199) *")" )

# ╔═╡ 21c3b05d-0134-4d5f-a7a3-822538be104c
begin
	FC2[ 0.0 .< FC2 .< 1.2e-2 ] .= 0.0
	FC2[ 0.0 .> FC2 .> -1.2e-2 ] .= 0.0
end

# ╔═╡ d0528b25-b21f-4c98-a029-5f8df0286f08
heatmap(1:size(FC2[:,:],1),
	    1:size(FC2[:,:],2), FC2[:,:],
	    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
	    xlabel="feature coord. (n x d)",
		ylabel="feature coord. (n x d)",
		aspectratio=:equal,
		size=(700, 700),
	    title="PbTe_FC2 (Traning Data = " *string(199) *")" )

# ╔═╡ 1dd85737-f044-4c59-bf4b-714dedd6629b
function recon_FC2(FC2)
	FC2_re = zeros(3,3,Int(size(FC2,1)/3),Int(size(FC2,1)/3));
	for i in 1:Int(size(FC2,1)/3)
		for j in 1:Int(size(FC2,1)/3)
			FC2_re[:,:,i,j] = FC2[3*(i-1)+1:3*i,3*(j-1)+1:3*j]
		end
	end
	return FC2_re
end

# ╔═╡ 81c979c7-f0a3-4cd4-bfc0-e47676a4ad14
FC2_re = recon_FC2(FC2);

# ╔═╡ bfed3725-ab73-4b0e-a673-4f26be3992d8
FC2_re[1:3,1:3,1,1:8]

# ╔═╡ cba31d94-4779-4db1-928f-5e58f83784f7
FC2_re[1:3,1:3,1,4]

# ╔═╡ 5e85a345-c83f-4f1c-b6bf-b93dd49500bc
FC2_re[1:3,1:3,1,6]

# ╔═╡ 8fe8b911-0e96-4159-8649-dddfdb896559
FC2_re[1:3,1:3,1,8]

# ╔═╡ 688e8cda-81cb-4729-8d90-a65a35f4cea0
FC2_re[1:3,1:3,1,10]

# ╔═╡ 1bef7de1-490e-4b01-8af3-ea9fd80c0075
FC2_re[1:3,1:3,1,12]

# ╔═╡ 7abb24bb-c752-4b13-b241-51a0b1617b8f
FC2_re[1:3,1:3,1,14]

# ╔═╡ 17d45498-853b-4aca-977d-cc46acf248d9
sum(FC2*equi)

# ╔═╡ 07d3592c-b5a6-49f5-92c9-093603ba42e1
sum(FC2)

# ╔═╡ e2b3d5cf-7f4f-4f22-b9f7-99d45b75166a
begin
	a1 = mean(
		[FC2_re[:,:,1,1][1,1] FC2_re[:,:,1,1][2,2] FC2_re[:,:,1,1][3,3]]
	)
	b1 = mean(
		[FC2_re[:,:,1,2][1,1] FC2_re[:,:,1,2][2,2] FC2_re[:,:,1,2][3,3]] 
	)
	c1 = mean(
		[FC2_re[:,:,1,4][3,3] FC2_re[:,:,1,6][2,2] FC2_re[:,:,1,8][1,1]
		 FC2_re[:,:,1,10][1,1] FC2_re[:,:,1,12][2,2] FC2_re[:,:,1,14][3,3]]	
	)
	d1 = mean(
		[FC2_re[:,:,1,3][1,2] FC2_re[:,:,1,3][2,1] 
		FC2_re[:,:,1,5][1,3] FC2_re[:,:,1,5][3,1] 
		-FC2_re[:,:,1,7][2,3] -FC2_re[:,:,1,7][3,2]
		FC2_re[:,:,1,9][2,3] FC2_re[:,:,1,9][3,2] 
		-FC2_re[:,:,1,11][1,3] -FC2_re[:,:,1,11][3,1] 
		-FC2_re[:,:,1,13][1,2] -FC2_re[:,:,1,13][2,1]]	
	)
	e1 = mean(
		[FC2_re[:,:,1,3][1,1] FC2_re[:,:,1,3][2,2] 
		FC2_re[:,:,1,5][1,1] FC2_re[:,:,1,5][3,3] 
		FC2_re[:,:,1,7][2,2] FC2_re[:,:,1,7][3,3]
		FC2_re[:,:,1,9][2,2] FC2_re[:,:,1,9][3,3] 
		FC2_re[:,:,1,11][1,1] FC2_re[:,:,1,11][3,3] 
		FC2_re[:,:,1,13][1,1] FC2_re[:,:,1,13][2,2]]	
	)
	f1 = mean(
		[FC2_re[:,:,1,3][3,3] FC2_re[:,:,1,5][2,2] FC2_re[:,:,1,7][1,1]
		 FC2_re[:,:,1,9][1,1] FC2_re[:,:,1,11][2,2] FC2_re[:,:,1,13][3,3]]	
	)
	g1 = mean(
		[FC2_re[:,:,1,15][1,1] FC2_re[:,:,1,15][2,2] FC2_re[:,:,1,15][3,3]]	
	)
	h1 = mean(
		[FC2_re[:,:,1,4][1,1] FC2_re[:,:,1,4][2,2] 
		FC2_re[:,:,1,6][1,1] FC2_re[:,:,1,6][3,3] 
		FC2_re[:,:,1,8][2,2] FC2_re[:,:,1,8][3,3]
		FC2_re[:,:,1,10][2,2] FC2_re[:,:,1,10][3,3] 
		FC2_re[:,:,1,12][1,1] FC2_re[:,:,1,12][3,3] 
		FC2_re[:,:,1,14][1,1] FC2_re[:,:,1,14][2,2]]	
	)
end


# ╔═╡ 71cc9b0b-7a8d-4425-9f09-8dae784d7fbb
FC2_re[:,:,2,11]

# ╔═╡ 8de1931a-f582-4d6a-be6d-650d85ae27c5
begin
	a2 = mean(
		[FC2_re[:,:,2,2][1,1] FC2_re[:,:,2,2][2,2] FC2_re[:,:,2,2][3,3]]
	)
	b2 = mean(
		[FC2_re[:,:,2,1][1,1] FC2_re[:,:,2,1][2,2] FC2_re[:,:,2,1][3,3]
		FC2_re[:,:,2,15][1,1] FC2_re[:,:,2,15][2,2] FC2_re[:,:,2,15][3,3]]
	)
	c2 = mean(
		[FC2_re[:,:,2,3][3,3] FC2_re[:,:,2,5][2,2] FC2_re[:,:,2,7][1,1]
		 FC2_re[:,:,2,9][1,1] FC2_re[:,:,2,11][2,2] FC2_re[:,:,2,13][3,3]]	
	)
	d2 = mean(
		[FC2_re[:,:,2,4][1,2] FC2_re[:,:,2,4][2,1] 
		FC2_re[:,:,2,6][1,3] FC2_re[:,:,2,6][3,1] 
		-FC2_re[:,:,2,8][2,3] -FC2_re[:,:,2,8][3,2]
		FC2_re[:,:,2,10][2,3] FC2_re[:,:,2,10][3,2] 
		-FC2_re[:,:,2,12][1,3] -FC2_re[:,:,2,12][3,1] 
		-FC2_re[:,:,2,14][1,2] -FC2_re[:,:,2,14][2,1]]	
	)
	e2 = mean(
		[FC2_re[:,:,2,4][1,1] FC2_re[:,:,2,4][2,2] 
		FC2_re[:,:,2,6][1,1] FC2_re[:,:,2,6][3,3] 
		FC2_re[:,:,2,8][2,2] FC2_re[:,:,2,8][3,3]
		FC2_re[:,:,2,10][2,2] FC2_re[:,:,2,10][3,3] 
		FC2_re[:,:,2,12][1,1] FC2_re[:,:,2,12][3,3] 
		FC2_re[:,:,2,14][1,1] FC2_re[:,:,2,14][2,2]]	
	)
	f2 = mean(
		[FC2_re[:,:,2,4][3,3] FC2_re[:,:,2,6][2,2] FC2_re[:,:,2,8][1,1]
		 FC2_re[:,:,2,10][1,1] FC2_re[:,:,2,12][2,2] FC2_re[:,:,2,14][3,3]]	
	)
	g2 = mean(
		[FC2_re[:,:,2,16][1,1] FC2_re[:,:,2,16][2,2] FC2_re[:,:,2,16][3,3]]	
	)
	h2 = mean(
		[FC2_re[:,:,2,3][1,1] FC2_re[:,:,2,3][2,2] 
		FC2_re[:,:,2,5][1,1] FC2_re[:,:,2,5][3,3] 
		FC2_re[:,:,2,7][2,2] FC2_re[:,:,2,7][3,3]
		FC2_re[:,:,2,9][2,2] FC2_re[:,:,2,9][3,3] 
		FC2_re[:,:,2,11][1,1] FC2_re[:,:,2,11][3,3] 
		FC2_re[:,:,2,13][1,1] FC2_re[:,:,2,13][2,2]]	
	)
end

# ╔═╡ 5fbb621a-bf58-4cdf-941c-5d277c7cfa5e
equi

# ╔═╡ 341e193a-587d-409d-9eb9-af3e2412f76b
g1

# ╔═╡ 710e8b0a-29ef-4caf-bf5f-0e8926add2c5
e2

# ╔═╡ 84cd6bd3-8834-4c82-873c-9d97f5c94b43
c2

# ╔═╡ 1333c23f-adaa-40a3-8d54-81339c564732
FC2_re[:,:,2,15]

# ╔═╡ Cell order:
# ╠═803d8357-aecc-41b0-ae4d-f2164f0e4e6b
# ╠═346974a0-8249-11ee-075d-33c092b30816
# ╠═d8b8b25d-e716-47e9-b2e4-12731df9e7c0
# ╠═39a72f49-6efb-4ad5-a3b0-7b6a6954d6a6
# ╠═e79b6a06-5285-44b2-81ea-e3a96247db8f
# ╠═e1d8fc9e-00fd-4fd9-975d-6b6a4f79068a
# ╠═a6a54c75-f12d-4694-a876-30f0a14ad8a1
# ╠═cbd94f02-b08d-4208-9e2a-b24a35d2646a
# ╠═e9efaf81-6634-435c-8030-45cc949d8068
# ╠═79d1e9f9-56d1-4425-9362-ff86a5c452f5
# ╠═25f5cb64-40e5-4a2a-9a0d-42878a3aa9ae
# ╠═b2ced1ff-11f5-46df-9e4a-a961ab8f0a63
# ╠═be8cba06-01f0-4ad5-861f-5a4c556a9cc4
# ╠═4451bffe-18c9-4da3-ba22-9df2cb2e6472
# ╠═21c3b05d-0134-4d5f-a7a3-822538be104c
# ╠═d0528b25-b21f-4c98-a029-5f8df0286f08
# ╠═1dd85737-f044-4c59-bf4b-714dedd6629b
# ╠═81c979c7-f0a3-4cd4-bfc0-e47676a4ad14
# ╠═bfed3725-ab73-4b0e-a673-4f26be3992d8
# ╠═cba31d94-4779-4db1-928f-5e58f83784f7
# ╠═5e85a345-c83f-4f1c-b6bf-b93dd49500bc
# ╠═8fe8b911-0e96-4159-8649-dddfdb896559
# ╠═688e8cda-81cb-4729-8d90-a65a35f4cea0
# ╠═1bef7de1-490e-4b01-8af3-ea9fd80c0075
# ╠═7abb24bb-c752-4b13-b241-51a0b1617b8f
# ╠═17d45498-853b-4aca-977d-cc46acf248d9
# ╠═07d3592c-b5a6-49f5-92c9-093603ba42e1
# ╠═e2b3d5cf-7f4f-4f22-b9f7-99d45b75166a
# ╠═71cc9b0b-7a8d-4425-9f09-8dae784d7fbb
# ╠═8de1931a-f582-4d6a-be6d-650d85ae27c5
# ╠═5fbb621a-bf58-4cdf-941c-5d277c7cfa5e
# ╠═341e193a-587d-409d-9eb9-af3e2412f76b
# ╠═710e8b0a-29ef-4caf-bf5f-0e8926add2c5
# ╠═84cd6bd3-8834-4c82-873c-9d97f5c94b43
# ╠═1333c23f-adaa-40a3-8d54-81339c564732
