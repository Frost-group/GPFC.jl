### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 3a4c7866-fa3d-11ed-382d-f971821a83fd
begin
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum
	using CSV
	using DataFrames
	using DelimitedFiles
	using Plots
end

# ╔═╡ 27fc8d3a-16ca-4143-916d-7351fab5aa4b
function kernelfunction(k, x₁, x₂::Vector{Float64}, grad::Int64)
	function f1st(x₁, x₂::Vector{Float64}) 
		Zygote.gradient( a -> k(a, x₂), x₁)[1]
	end	
	function f2nd(x₁, x₂::Vector{Float64})
		Zygote.hessian(a -> k(a, x₂), x₁)
	end
	function f3rd(x₁, x₂::Vector{Float64}) 
		ForwardDiff.jacobian( a -> f2nd(a, x₂), x₁)
	end 
	function f4th(x₁, x₂::Vector{Float64})
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

# ╔═╡ fa5ec98c-99bc-4a07-a398-ce14043e5d20
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

# ╔═╡ 1da36f99-7e15-48e3-ae08-2da5769cb506
function ph_trans(equi, feature, force, energy, phase, mass, eigVec)
	equi_ph = eigVec' * mass * 1/sqrt(2) * phase * (equi)
	Nor_dim = size(mass, 1)
	Num_data = size(feature, 2)
	dim = size(feature, 1)
	feature_ph = zeros((Nor_dim, Num_data))
	force_ph = zeros((Nor_dim*Num_data, 1))

	for ii in 1:Num_data
		feature_ph[:, ii] = eigVec' * mass * 1/sqrt(2) * phase * (feature[:, ii])
		force_ph[Nor_dim*(ii-1)+1 : Nor_dim*ii] = eigVec' * mass * 1/sqrt(2) * phase * (force[dim*(ii-1)+1 : dim*ii]-force[1:dim])
	end

	Target_ph = vcat(energy, reshape(force_ph, (Nor_dim*Num_data,1)))
	
	return equi_ph, feature_ph, Target_ph
end

# ╔═╡ 6eb33236-3afc-4c2e-a004-b393bc4946c8
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

# ╔═╡ 2e2fb6be-646e-41ff-83bb-69dd12f81ff4
function Coveriance_energy(X::Matrix{Float64}, xₒ::Vector{Float64}, k)
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for Energy prediction
	#building Covariance matrix containers
	K₀ₙₘ= zeros(((1+dim)*num))
	for j in 1:num
		#Fillin convarian of Energy vs Energy
		K₀ₙₘ[j] = kernelfunction(k, X[:,j], xₒ, 0)
		#Fillin convarian of Force vs Energy
		K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =  kernelfunction(k, X[:,j], xₒ, 1)
	end
	return K₀ₙₘ
end

# ╔═╡ 7c1d2c93-bc41-4144-bfb0-2128ce3519e9
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

# ╔═╡ 8eb52208-b603-48dc-9c54-84f35f669b65
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

# ╔═╡ e4de5b20-3d5f-44c7-a696-a44e1de9341e
begin
	Featurefile = "feature_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
	Energyfile = "energy_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
	Forcefile = "force_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv"
end;

# ╔═╡ 55c30380-d9ef-41e9-9dae-9700fe17e3f5
begin
	eigVecG = 
[[ 0.00000000e+00 -7.07106781e-01  0.00000000e+00 0.00000000e+00 -7.07106781e-01  0.00000000e+00]
[ 5.73226443e-01  0.00000000e+00  4.14018653e-01 -1.01875558e-02  0.00000000e+00  7.07033389e-01]
[-4.14018653e-01  0.00000000e+00  5.73226443e-01 7.07033389e-01  0.00000000e+00  1.01875558e-02]
[ 0.00000000e+00 -7.07106781e-01  0.00000000e+00 -5.20417043e-18  7.07106781e-01  0.00000000e+00]
[ 5.73226443e-01 -5.98176036e-18  4.14018653e-01 1.01875558e-02  5.98176036e-18 -7.07033389e-01]
[-4.14018653e-01  5.98176036e-18  5.73226443e-01 -7.07033389e-01 -5.98176036e-18 -1.01875558e-02]];
	dim = 3
	nump = 2
	nums = 16
	amu = 1
	mass = sqrt(28.085*amu)* Matrix(I , dim*nump, dim*nump)
end;

# ╔═╡ 63473888-5aeb-4d48-8020-27e06147de3f
Dyna = [[ 4.67251785e-01  0.00000000e+00  0.00000000e+00 -4.67287390e-01 -3.95301143e-18  3.95301143e-18]
       [ 0.00000000e+00  4.67251785e-01  0.00000000e+00 -3.95301143e-18 -4.67287390e-01  3.95301143e-18]
       [ 0.00000000e+00  0.00000000e+00  4.67251785e-01 3.95301143e-18  3.95301143e-18 -4.67287390e-01]
       [-4.67287390e-01 -3.95301143e-18  3.95301143e-18 4.67251785e-01  0.00000000e+00  0.00000000e+00]
       [-3.95301143e-18 -4.67287390e-01  3.95301143e-18 0.00000000e+00  4.67251785e-01  0.00000000e+00]
       [ 3.95301143e-18  3.95301143e-18 -4.67287390e-01 0.00000000e+00  0.00000000e+00  4.67251785e-01]];

# ╔═╡ 27b8d359-5a0f-4d7a-bc70-3eb02e1d8c87
# ╠═╡ disabled = true
#=╠═╡
A = eigVecG' * Dyna * eigVecG
  ╠═╡ =#

# ╔═╡ 2b4112df-db79-48ee-9579-4961854511b2
#=╠═╡
heatmap(1:6,1:6, A)
  ╠═╡ =#

# ╔═╡ dda9dfc0-4581-4a93-aa48-e0fe61c5d0f2
begin
	data = eigen(Dyna)
	eigVecg = data.vectors
	eigValg = data.values
end

# ╔═╡ 06a9ec80-ba5e-4587-b448-2c87c0178aad
begin
	σₒ = 1.0                   # Kernel Scale
	l = 1.0				    # Length Scale
	lph = det(eigVecg)* l * sqrt(28.085/2) 
	σₑ = 1e-5 					# Energy Gaussian noise
	σₙ = 1e-6                   # Force Gaussian noise for Model 2 (σₑ independent)
		
	Num = 100                   # Number of training points
	DIM = 3                     # Dimension of Materials
	model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
	order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
		
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
	kernelph = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(lph)
end;

# ╔═╡ 3259213b-f48d-4209-9cf1-cfe9ff6ef795
equi, feature, energy, force, Target = ASEFeatureTarget(
    Featurefile, Energyfile, Forcefile, Num, DIM);

# ╔═╡ b28e2e9b-8a94-4229-af3f-5b6a574e96fc
begin
	qpointG = [0. 0. 0.];
	
	phaseG = zeros((dim*nump, dim*nums))
	for  ii in 1:nums
		for jj in 1:nump
			if mod(ii, nump) == mod(jj, nump) 
				phaseG[3*(jj-1)+1:3*jj, 3*(ii-1)+1:3*ii] = 
				Matrix(I, 3, 3) * exp(-dot(qpointG, equi[3*(ii-1)+1:3*ii]) * 1im)
			end
		end
	end
end;

# ╔═╡ 730e31a0-418a-4155-a2d3-218a5fe0102a
heatmap(1:6,1:6,eigVecg)

# ╔═╡ 4972e5bf-9742-491d-9165-b9c82d0b39b3
eigVecg' * Dyna * eigVecg

# ╔═╡ 67442a55-ae04-4cc9-ba54-05fcce62c0b8
eigVecg

# ╔═╡ 9563fef3-073d-43e1-b468-bf9b7aab99de
 Dyna * eigVecg[4,1:6] ==  3.5605e-5 * eigVecg[4,1:6]

# ╔═╡ 8c2dd257-01dc-4ac7-8e0e-620cebf06c35
equi_ph, feature_ph, Target_ph = ph_trans(equi, feature, force, energy, phaseG, mass, eigVecg);

# ╔═╡ ccdfed5d-479a-49c9-822a-5c90c9db6ff2
@time Kmm_ph = Marginal(feature_ph, kernelph, lph, σₑ, σₙ);

# ╔═╡ d426b2c4-5c30-40cf-920b-a75d4a11d4db
 @time K₀ₙₘph = Coveriance_energy(feature_ph, equi_ph, kernelph);

# ╔═╡ 43f36150-6e48-4401-887d-8ccd1170d16a
@time Posterior(Kmm_ph, K₀ₙₘph, Target_ph)

# ╔═╡ bc0e4c4c-8fee-4233-addd-f85f4aea49fb
@time K₂ₙₘ = Coveriance_fc2(feature_ph, equi_ph, kernelph);

# ╔═╡ e9d8c24f-4b76-4ac6-adb6-166b1681253b
@time FC2_ph = Posterior(Kmm_ph, K₂ₙₘ, Target_ph);

# ╔═╡ 523296ca-3ac4-4c54-9872-4ba77f2bdc13
begin
	function Car2Ph_1st(phase, mass, eigvec, Tensor1st)
		transMat = eigvec' * inv(mass)/sqrt(2) * phase
		return transMat * Tensor1st
	end
	function Car2Ph_2nd(phase, mass, eigvec, Tensor2nd)
		transMat = eigvec' * inv(mass)/sqrt(2) * phase
		return transMat * Tensor2nd * transMat'
	end
end;

# ╔═╡ ab4fd18b-1918-4329-baf4-0f5ad3a954c7
function Marginal_ph(X, k, l, σₑ, σₙ, phase, mass, eigvec)
	dim = Int(size(feature,1)/8)
	num = size(X,2)
	#building Marginal Likelihood containers
	#For Energy + Force
	KK = zeros(((1+dim)*num, (1+dim)*num))
	
	for i in 1:num 
		for j in 1:num 
		#Fillin convarian of Energy vs Energy
			KK[i, j] = kernelfunction(k, X[:,i], X[:,j], 0)
		#Fillin convarian of Force vs Energy
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = Car2Ph_1st(phase, mass, eigvec, kernelfunction(k, X[:,i], X[:,j], 1))
		#Fillin convarian of Energy vs Force	
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j]
		#Fillin convarian of Energy vs Force
			KK[(num+1)+((i-1) * dim):(num+1)+((i) * dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = Car2Ph_2nd(phase, mass, eigvec, -kernelfunction(k, X[:,i], X[:,j], 2))
		end
	end

	Iee = σₑ^2 * Matrix(I, num, num)
	Iff = (σₑ / l)^2 * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

	Kₘₘ = KK + II
	
	return Kₘₘ
end

# ╔═╡ aae137f6-a793-4c07-a7a9-0f668b3bacc6
function Coveriance_energy_ph(X, xₒ, k, phase, mass, eigvec)
	dim = Int(size(X,1)/8)
	num = size(X,2)
	
	#Covariance matrix for Energy prediction
	#building Covariance matrix containers
	K₀ₙₘ= zeros(((1+dim)*num))
	for j in 1:num
		#Fillin convarian of Energy vs Energy
		K₀ₙₘ[j] = kernelfunction(k, X[:,j], xₒ, 0)
		#Fillin convarian of Force vs Energy
		K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =  Car2Ph_1st(phase, mass, eigvec, kernelfunction(k, X[:,j], xₒ, 1))
	end
	return K₀ₙₘ
end

# ╔═╡ 5fb20402-fd3b-4f69-bfca-585e807959da


# ╔═╡ 1f748b0c-2969-4f01-95db-271b205cdb28
begin
	Mph = Marginal_ph(feature, kernel, l, σₑ, σₙ, phaseG, mass, eigVecg)
	heatmap(1:700,1:700,Mph)
end

# ╔═╡ 51dbe12d-974f-453b-9515-3b02f0c5d932
begin
	M = Marginal(feature_ph, kernelph, l, σₑ, σₙ)
	heatmap(1:700,1:700,M)
end

# ╔═╡ 1d4d4eeb-659f-439c-b008-f087749f0497
begin
	Cph = Coveriance_energy_ph(feature, equi, kernel, phaseG, mass, eigVecg)
end

# ╔═╡ 20344865-935a-43d0-a8c3-f0e0a9690dfb
begin
	C = Coveriance_energy(feature_ph, equi_ph, kernelph)
end

# ╔═╡ 3151bfc3-ae98-4c60-b1d4-f65c4f8c965e
energy[1]

# ╔═╡ c2a93149-3fb2-4ceb-b487-15cb051a8fdc
@time Posterior(Mph, Cph, Target_ph)

# ╔═╡ c2f0e04a-5d7a-46b9-a3ed-74885c052ccb
@time Posterior(M, C, Target_ph)

# ╔═╡ c36344d7-80e7-48e6-b608-5fe97d492209
begin
	Ecar = zeros((9))
	Eph = zeros((9))
	nd = [2, 5 , 10, 15, 20, 30, 50, 70, 100]
	Ecar2 = zeros((9))
	Eph2 = zeros((9))
end;

# ╔═╡ 056d0f85-316b-49c7-bfc9-b6167bb3e676
@time for k in 1:9
	numt1 = nd[k]
	equi, feature, energy, force, Target = ASEFeatureTarget(
    Featurefile, Energyfile, Forcefile, numt1, DIM);

	
	equi_ph, feature_ph, Target_ph = ph_trans(equi, feature, force, energy, phaseG, mass, eigVecg);
	
	Kₘₘ = Marginal_ph(feature, kernel, l, σₑ, σₙ, phaseG, mass, eigVecg);
	K₀ₙₘ = Coveriance_energy_ph(feature, equi, kernel, phaseG, mass, eigVecg);

	
	Kₘₘph = Marginal(feature_ph, kernelph, lph, σₑ, σₙ);
	K₀ₙₘph = Coveriance_energy(feature_ph, equi_ph, kernelph);


	Ecar[k] = abs(Posterior(Kₘₘ, K₀ₙₘ, Target_ph)-Target[1])
	Eph[k] = abs(Posterior(Kₘₘph, K₀ₙₘph, Target_ph)-Target[1])

end

# ╔═╡ d991e850-1759-4976-924f-ac99524328f0
anim = @animate for i in 1:9
	plot(nd[1:i], [Ecar[1:i], Eph[1:i]],
		xlabel="Training points",
		ylabel="Error",
		xlim = (-1, 110), 
		ylim = (-1e-5, 1e-4),
		labels = ["Cartesian" "Phonon"],
		linewidth=3,
		title="PES Error (Traning Data = " *string(nd[i]) *")"
	)
end

# ╔═╡ 0450c18d-1a0d-4285-84f1-c41962213c0a
gif(anim, "Si_Energy_ph_02062023.gif", fps=2)

# ╔═╡ 7b14452e-65a3-42cb-beed-89d01d57432a
@time for k in 1:9
	numt1 = nd[k]
	equi, feature, energy, force, Target = ASEFeatureTarget(
    Featurefile, Energyfile, Forcefile, numt1, DIM);

	feature2 = feature[:, 2:numt1];
	force2 = force[49:48*numt1];
	energy2 = energy[2:numt1];
	Target2 = zeros(((48+1)*(numt1-1),1));
	Target2[1:numt1-1,:] = energy2;
	Target2[numt1:(numt1-1)*(1+48),:] = force2;
	
	equi_ph, feature_ph, Target_ph = ph_trans(equi, feature2, force2, energy2, phaseG, mass, eigVecg);

	
	Kₘₘ = Marginal_ph(feature2, kernel, l, σₑ, σₙ, phaseG, mass, eigVecg);
	K₀ₙₘ = Coveriance_energy_ph(feature2, equi, kernel, phaseG, mass, eigVecg);

	
	Kₘₘph = Marginal(feature_ph, kernelph, lph, σₑ, σₙ);
	K₀ₙₘph = Coveriance_energy(feature_ph, equi_ph, kernelph);


	Ecar2[k] = abs(Posterior(Kₘₘ, K₀ₙₘ, Target_ph)-Target[1])
	Eph2[k] = abs(Posterior(Kₘₘph, K₀ₙₘph, Target_ph)-Target[1])
end

# ╔═╡ 7220f0a4-1b8a-4f07-89c1-88558073b3c0
anim2 = @animate for i in 1:9
	plot(nd[1:i], [Ecar2[1:i], Eph2[1:i]],
		xlabel="Training points",
		ylabel="Error",
		xlim = (-1, 110), 
		ylim = (-1e-5, 1e-0),
		labels = ["Cartesian" "Phonon"],
		linewidth=3,
		title="PES Error (Traning Data = " *string(nd[i]) *")"
	)
end

# ╔═╡ e0d04409-f77d-4076-90e0-5bd692799b43
gif(anim2, "Si_Energy_ph2_020620230.gif", fps=2)

# ╔═╡ 96439fe8-7f62-4676-a8fd-72b6a6f5e4a7
begin
	dis = LinRange(-0.02, 0.02, 11)
	Ecar_dis = zeros((11))
	Eph_dis = zeros((11))
	Ecar2ph_dis = zeros((11))
end

# ╔═╡ da913a95-a262-4193-9cd9-d402ffcad68e

begin
	DIS = zeros((48))
		for kk in 1:16
			if mod(kk,2) == 1
				DIS[3*(kk-1)+1:3*kk] = [1 1 1]
			end
		end
	end

# ╔═╡ c37e43af-dcf9-4b78-a0b6-c546d2d0fba3
begin
	KₘₘCar = Marginal(feature, kernel, l, σₑ, σₙ);
	Kₘₘph = Marginal(feature_ph, kernelph, lph, σₑ, σₙ);
	KₘₘCar2ph = Marginal_ph(feature, kernel, l, σₑ, σₙ, phaseG, mass, eigVecG);
	@time for k in 1:11
		equik = equi + (DIS * dis[k])
		equik_ph = (eigVecg' * inv(mass)/sqrt(2) * phaseG * equi) + dis[k]*[1, 1, 1, 0, 0, 0] * sqrt(28.085/2)
		
		K₀ₙₘCar = Coveriance_energy(feature, equik, kernel);
		K₀ₙₘph = Coveriance_energy(feature_ph, equik_ph, kernelph);
		K₀ₙₘCar2ph = Coveriance_energy_ph(feature, equik, kernel, phaseG, mass, eigVecg);
		
		Ecar_dis[k] = Posterior(KₘₘCar, K₀ₙₘCar, Target)
		Eph_dis[k] = Posterior(Kₘₘph, K₀ₙₘph, Target_ph)
		Ecar2ph_dis[k] = Posterior(KₘₘCar2ph, K₀ₙₘCar2ph, Target_ph)
	end
end

# ╔═╡ a9fe24cf-7daa-45ba-9666-321146e0caa4
plot(dis[1:11], [Ecar_dis[1:11], Eph_dis[1:11], Ecar2ph_dis[1:11]],
		xlabel="Displacement in Angstom",
		ylabel="Harmonic PES",
		#xlim = (-1, 110), 
		ylim = (-86.5, -86),
		labels = ["Cartesian" "Ph bf Deri" "Deri bf Ph"],
		linewidth=3,
		title="Harmonic PES by shaking 1st Si"
	)

# ╔═╡ 5da64d73-64fc-4da1-811a-73aee237c3be
Eph_dis

# ╔═╡ 3ebcbb90-6312-446d-b75a-9b77803432b8
 Ecar2ph_dis

# ╔═╡ dcc3f238-8d61-4b7c-9924-c3ec415e8cff
eigVecg' * inv(mass)/sqrt(2) * phaseG * (equi) + dis[1]*[1, 1, 1, 0, 0, 0] * sqrt(28.085/2)

# ╔═╡ 56dc3a55-a8a8-4727-b4d6-0460c1a93f82
eigVecg' * inv(mass)/sqrt(2) * phaseG * (DIS * dis[1])

# ╔═╡ Cell order:
# ╠═3a4c7866-fa3d-11ed-382d-f971821a83fd
# ╠═27fc8d3a-16ca-4143-916d-7351fab5aa4b
# ╠═fa5ec98c-99bc-4a07-a398-ce14043e5d20
# ╠═1da36f99-7e15-48e3-ae08-2da5769cb506
# ╠═6eb33236-3afc-4c2e-a004-b393bc4946c8
# ╠═2e2fb6be-646e-41ff-83bb-69dd12f81ff4
# ╠═7c1d2c93-bc41-4144-bfb0-2128ce3519e9
# ╠═8eb52208-b603-48dc-9c54-84f35f669b65
# ╠═e4de5b20-3d5f-44c7-a696-a44e1de9341e
# ╠═06a9ec80-ba5e-4587-b448-2c87c0178aad
# ╠═3259213b-f48d-4209-9cf1-cfe9ff6ef795
# ╠═55c30380-d9ef-41e9-9dae-9700fe17e3f5
# ╠═63473888-5aeb-4d48-8020-27e06147de3f
# ╠═27b8d359-5a0f-4d7a-bc70-3eb02e1d8c87
# ╠═2b4112df-db79-48ee-9579-4961854511b2
# ╠═b28e2e9b-8a94-4229-af3f-5b6a574e96fc
# ╠═dda9dfc0-4581-4a93-aa48-e0fe61c5d0f2
# ╠═730e31a0-418a-4155-a2d3-218a5fe0102a
# ╠═4972e5bf-9742-491d-9165-b9c82d0b39b3
# ╠═67442a55-ae04-4cc9-ba54-05fcce62c0b8
# ╠═9563fef3-073d-43e1-b468-bf9b7aab99de
# ╠═8c2dd257-01dc-4ac7-8e0e-620cebf06c35
# ╠═ccdfed5d-479a-49c9-822a-5c90c9db6ff2
# ╠═d426b2c4-5c30-40cf-920b-a75d4a11d4db
# ╠═43f36150-6e48-4401-887d-8ccd1170d16a
# ╠═bc0e4c4c-8fee-4233-addd-f85f4aea49fb
# ╠═e9d8c24f-4b76-4ac6-adb6-166b1681253b
# ╠═523296ca-3ac4-4c54-9872-4ba77f2bdc13
# ╠═ab4fd18b-1918-4329-baf4-0f5ad3a954c7
# ╠═aae137f6-a793-4c07-a7a9-0f668b3bacc6
# ╠═5fb20402-fd3b-4f69-bfca-585e807959da
# ╠═1f748b0c-2969-4f01-95db-271b205cdb28
# ╠═51dbe12d-974f-453b-9515-3b02f0c5d932
# ╠═1d4d4eeb-659f-439c-b008-f087749f0497
# ╠═20344865-935a-43d0-a8c3-f0e0a9690dfb
# ╠═3151bfc3-ae98-4c60-b1d4-f65c4f8c965e
# ╠═c2a93149-3fb2-4ceb-b487-15cb051a8fdc
# ╠═c2f0e04a-5d7a-46b9-a3ed-74885c052ccb
# ╠═c36344d7-80e7-48e6-b608-5fe97d492209
# ╠═056d0f85-316b-49c7-bfc9-b6167bb3e676
# ╠═d991e850-1759-4976-924f-ac99524328f0
# ╠═0450c18d-1a0d-4285-84f1-c41962213c0a
# ╠═7b14452e-65a3-42cb-beed-89d01d57432a
# ╠═7220f0a4-1b8a-4f07-89c1-88558073b3c0
# ╠═e0d04409-f77d-4076-90e0-5bd692799b43
# ╠═96439fe8-7f62-4676-a8fd-72b6a6f5e4a7
# ╠═da913a95-a262-4193-9cd9-d402ffcad68e
# ╠═c37e43af-dcf9-4b78-a0b6-c546d2d0fba3
# ╠═a9fe24cf-7daa-45ba-9666-321146e0caa4
# ╠═5da64d73-64fc-4da1-811a-73aee237c3be
# ╠═3ebcbb90-6312-446d-b75a-9b77803432b8
# ╠═dcc3f238-8d61-4b7c-9924-c3ec415e8cff
# ╠═56dc3a55-a8a8-4727-b4d6-0460c1a93f82
