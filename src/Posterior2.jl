function kernelfunction(k, x₁, x₂::Vector{Float64}, grad::Int64)
	function f1st(x₁, x₂::Vector{Float64}) 
		ForwardDiff.gradient( a -> k(a, x₂), x₁)
	end	
	function f2nd(x₁, x₂::Vector{Float64})
		ForwardDiff.jacobian( a -> f1st(a, x₂), x₁)
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


function Marginal2(X::Matrix{Float64}, k, l::Float64, σₑ::Float64, σₙ::Float64, model::Int64)
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
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -kernelfunction(k, X[:,i], X[:,j], 1)
		#Fillin convarian of Energy vs Force
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = -kernelfunction(k, X[:,i], X[:,j], 2)
		#For traning on Energy and Force separately
			K₀₀[i, j] = KK[i, j]
			
			K₁₁[(1)+((i-1)*dim):(1)+((i)*dim)-1,
				(1)+((j-1)*dim):(1)+((j)*dim)-1] = KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1, (num+1)+((j-1)*dim):(num+1)+((j)*dim)-1]
		end
	end

#Gaussian noise model
	# First model the noise for Energy relating to the Force noise by l⁻² 
	if model == 1
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = (σₑ / l)^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
		
	# Second model the both noises are independent	
	else
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
	end
	
	return K₀₀, K₁₁, Kₘₘ
end

function Coveriant2(X::Matrix{Float64}, xₒ::Vector{Float64}, k, order::Int64)
	dim = size(X,1)
	num = size(X,2)
	
	
	#Covariance matrix for Energy prediction
	if order == 0
		#building Covariance matrix containers
		K₀ₙₘ= zeros(((1+dim)*num))
		for j in 1:num
			
			#Fillin convarian of Energy vs Energy
			K₀ₙₘ[j] = kernelfunction(k, X[:,j], xₒ, 0)
			#Fillin convarian of Force vs Energy
			K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =  kernelfunction(k, X[:,j], xₒ, 1)
		end
		Kₙₘ = K₀ₙₘ
		
	#Covariance matrix for Force prediction	
	elseif order == 1
		#building Covariance matrix containers
		K₁ₙₘ= zeros((dim, (1+dim)*num))
		
		for j in 1:num
			
			#Fillin convarian of Energy vs Force
			K₁ₙₘ[:,j] = 
				reshape(
					-  kernelfunction(k, X[:,j], xₒ, 1)
				, (dim)
			)
			#Fillin convarian of Force vs Force
			K₁ₙₘ[:, (num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					-  kernelfunction(k, X[:,j], xₒ, 2)
					, (dim, dim)
				)
		end
		Kₙₘ = K₁ₙₘ  
		
	#Covariance matrix for FC2 prediction
	elseif order == 2
		
		K₂ₙₘ= zeros((dim, dim, (1+dim)*num))
		
		for j in 1:num
			#Fillin convarian of Energy vs FC2
			K₂ₙₘ[:,:,j] = 
				reshape(
					 kernelfunction(k, X[:,j], xₒ, 2)
					, (dim, dim)
				)
			#Fillin convarian of Force vs FC2
			K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					 kernelfunction(k, X[:,j], xₒ, 3)
					, (dim, dim, dim)
				)
		end
		Kₙₘ = K₂ₙₘ
		
	#Covariance matrix for FC3 prediction
	elseif order == 3
		#building Covariance matrix containers
		K₃ₙₘ= zeros((dim, dim, dim, (1+dim)*num))
		
		for j in 1:num
			#Fillin convarian of Energy vs FC3
			K₃ₙₘ[:,:,:,j] = 
				reshape(
					-  kernelfunction(k, X[:,j], xₒ, 3)
					, (dim, dim, dim)
				)
			#Fillin convarian of Force vs FC3
			K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					-  kernelfunction(k, X[:,j], xₒ, 4)
					, (dim, dim, dim, dim)
				)
		end
		Kₙₘ = K₃ₙₘ
	end

	return Kₙₘ
end

function PosteriorMean2(X::Matrix{Float64}, xₒ::Vector{Float64}, Target::Matrix{Float64}, k, l::Float64, σₑ::Float64, σₙ::Float64, order::Int64, model::Int64)
	dim = size(X,1)
	num = size(X,2)
	K₀₀, K₁₁, Kₘₘ = Marginal2(X, k, l, σₑ, σₙ, model )
	Kₙₘ = Coveriant2(X, xₒ, k, order)
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