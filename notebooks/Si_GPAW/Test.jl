begin
	using KernelFunctions
	using ForwardDiff
	using LinearAlgebra
	using Einsum
	using CSV
	using DataFrames
	using Optim
	using Distributed
end

function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, numt, dimA)
	a  = 4 - dimA
	feature = (
		CSV.File(
			FileFeature
		)|> Tables.matrix
	)[
		begin:a:end
		,2:numt+1
	]
	equi = feature[:,1]
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (
		CSV.File(
			FileEnergy
		)|> Tables.matrix
	)[
		begin:numt
		,2
	]

	force = -reshape(
	(
		CSV.File(
			FileForce
		)|> Tables.matrix
	)[
		begin:a:end
		,2:numt+1
	]
	, (dim*num,1)
	)

	Target = vcat(
		energy
		, reshape(
			force
			, (dim*num,1)
		)
	)

	return equi, feature, energy, force, Target
end

function kernel(k, xₜ, vₜ, grad)

#order 0
	if grad == [0,0]
		return k(xₜ, vₜ)

#order 1
	elseif grad == [1,0]
		return   ForwardDiff.gradient(
			x -> k(x, vₜ)
			, xₜ)

	elseif grad == [0,1]
		return - ForwardDiff.gradient(
			x -> k(x, vₜ)
			, xₜ)

#order 2		

	elseif grad == [1,1]
		return - ForwardDiff.jacobian(
			x -> ForwardDiff.gradient(
				x -> k(x, vₜ)
				, x)
			, xₜ)

	elseif grad == [2,0] || grad == [0,2]
		return   ForwardDiff.jacobian(
			x -> ForwardDiff.gradient(
				x -> k(x, vₜ)
				, x)
			, xₜ)

#order 3		
	elseif grad == [3,0] || grad == [1,2]
		return   ForwardDiff.jacobian(
		x -> ForwardDiff.jacobian(
			x -> ForwardDiff.gradient(
				x -> k(x, vₜ)
				, x)
			, x)
		, xₜ)

	elseif grad == [2,1] || grad == [0,3]
		return - ForwardDiff.jacobian(
		x -> ForwardDiff.jacobian(
			x -> ForwardDiff.gradient(
				x -> k(x, vₜ)
				, x)
			, x)
		, xₜ)

#order 4
	elseif grad == [4,0] || grad == [2,2] || grad == [0,4]
		return   ForwardDiff.jacobian(
		x ->ForwardDiff.jacobian(
			x -> ForwardDiff.jacobian(
				x -> ForwardDiff.gradient(
					x -> k(x, vₜ)
					, x)
				, x)
			, x)
		, xₜ)

	elseif grad == [3,1] || grad == [1,3] 
		return - ForwardDiff.jacobian(
		x ->ForwardDiff.jacobian(
			x -> ForwardDiff.jacobian(
				x -> ForwardDiff.gradient(
					x -> k(x, vₜ)
					, x)
				, x)
			, x)
		, xₜ)

	end
end

function Marginal(X, k, σₑ, σₙ; model = 1)
	dim = size(X,1)
	num = size(X,2)
	KK = zeros(
		(
			(1+dim)*num, (1+dim)*num
		)
	)
	K₀₀ = zeros(
		(
			(1)*num, (1)*num
		)
	)
	K₁₁ = zeros(
		(
			(dim)*num, (dim)*num
		)
	)
	
	for i in 1:num 
		for j in 1:num 
			KK[i, j] = kernel(
				k, X[:,i], X[:,j], [0,0]
			)
			
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernel(
				k, X[:,i], X[:,j], [1,0]
			)
			
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = kernel(
				k, X[:,i], X[:,j], [0,1]
			)
			
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,
				(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = kernel(
					k, X[:,i], X[:,j], [1,1]
				)
			
			K₀₀[i, j] = KK[i, j]
			K₁₁[(1)+((i-1)*dim):(1)+((i)*dim)-1,
				(1)+((j-1)*dim):(1)+((j)*dim)-1] = KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1, 
(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1]
			
		end
	end
	
	if model == 1
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = (σₑ / l)^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
	else
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
	end
	#Kₘₘ⁻¹ = inv(KK+II)
	return K₀₀, K₁₁, Kₘₘ
end



function Coveriant(X, xₒ, k, order)
	dim = size(X,1)
	num = size(X,2)

	Kₙₘ= zeros(
		(
			(dim^order), (1+dim)*num
		)
	)
	if order == 0
		K₀ₙₘ= zeros(
			((1+dim)*num)
		)
		for j in 1:num 
			K₀ₙₘ[j] = 
				kernel(
					k, X[:,j], xₒ, [0,0]
			)
			K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				kernel(
					k, X[:,j], xₒ, [1,0]
				)
		end
		Kₙₘ = K₀ₙₘ
		
	elseif order == 1
		K₁ₙₘ= zeros(
			(dim, (1+dim)*num)
		)
		for j in 1:num 
			K₁ₙₘ[:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,1]
					), (dim)
				)
			K₁ₙₘ[:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,1]
					), (dim, dim)
				)
		end
		Kₙₘ = K₁ₙₘ  
	
	elseif order == 2
		K₂ₙₘ= zeros(
			(dim, dim, (1+dim)*num)
		)
		for j in 1:num 
			K₂ₙₘ[:,:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,2]
					), (dim, dim)
				)
			K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,2]
					), (dim, dim, dim)
				)
		end
		Kₙₘ = K₂ₙₘ

	elseif order == 3
		K₃ₙₘ= zeros(
			(dim, dim, dim, (1+dim)*num)
		) 
		for j in 1:num 
			K₃ₙₘ[:,:,:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,3]
					), (dim, dim, dim)
				)
			K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,3]
					), (dim, dim, dim, dim)
				)
		end
		Kₙₘ = K₃ₙₘ
	end

	return Kₙₘ
end

function Posterior(X, xₒ, Target, k, σₑ, σₙ, order, model)
	dim = size(X,1)
	num = size(X,2)
	K₀₀, K₁₁, Kₘₘ = Marginal(X, k, σₑ, σₙ; model )
	Kₙₘ = Coveriant(X, xₒ, k, order)
	Kₘₘ⁻¹ = inv(Kₘₘ)
	
	if order == 0
		Meanₚ = Kₙₘ' * Kₘₘ⁻¹ * Target
		
	elseif order == 1
		Meanₚ = ones(dim)
		@einsum Meanₚ[i] = Kₙₘ[i, m] * Kₘₘ⁻¹[m, n] * Target[n]
	
	elseif order == 2
		Meanₚ = ones(dim, dim)
		@einsum Meanₚ[i, j] = Kₙₘ[i, j, m] * Kₘₘ⁻¹[m, n] * Target[n]

	elseif order == 3
		Meanₚ = ones(dim, dim, dim)
		@einsum Meanₚ[i, j, k] = Kₙₘ[i, j, k, m] * Kₘₘ⁻¹[m, n] * Target[n]
	end
	
	return Meanₚ, K₀₀, K₁₁, Kₘₘ, Kₙₘ
end

begin
	#σₒ = 0.1
	#l = 0.4
	#σₑ = 0.00001
	#numt = 48
	σₒ = 0.1
	l = 0.4
	σₑ = 0.00001
	
	σₙ = 0.000001
	DIM = 3
	model = 1
	order = 2
	
	kₛₑ2 = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
end


begin
	numt = 1
	equiSi, featureSi, energySi, forceSi, TargetSi = ASEFeatureTarget(
				"feature_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv",
				"energy_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv", 
				"force_Si_222spc_01_n100_PW800_kpts9_e100_d1.csv", numt, DIM)
	FC_Si, K₀₀Si, K₁₁Si, KₘₘSi, KₙₘSi = Posterior(featureSi, equiSi, TargetSi, kₛₑ2, σₑ, σₙ, order, model)
	FC_Si
end
