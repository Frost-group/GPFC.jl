"""
    kernelfunction(k, x₁, x₂::Vector{Float64}, grad::Int64)

Calculate the derivative of kernel function 'k'.
A standard (or compositing) kernel function requires to be defined by using KernelFunctions.jl.
This will turn the defined kernel function and two different atomistic
representation vectors (features) to the derivative of the kernel of those two vectors where the
derivative order can be specified by variable grad. 
"""
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

    # JMF: ToDo - should be able to rewrite this as some kind of multiple dispatch
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

"""
	Marginal(X::Matrix{Float64}, k, l::Float64, σₑ::Float64)

Construct Marginal likelihood matrix of a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
through a standard (or compositing) kernel function 'k'.
Marginal() function alsom require l (model length scale) and σₑ (scaling factor) used to evalua
ate a model Gaussian noise.
"""

function Marginal(X::Matrix{Float64}, k, l::Float64, σₑ::Float64)
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
	Iff = -(σₑ / l)^2 * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

	Kₘₘ = KK + II
	
	return Kₘₘ
end

"""
Covariance class
"""
"""
	Coveriance_energy(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function 'k') between a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the potential energy surface at the equilibrium point.
"""

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

"""
	Coveriance_force(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function 'k') between a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the force fields (which should be zeros) at the equilibrium point.
"""

function Coveriance_force(X::Matrix{Float64}, xₒ::Vector{Float64}, k)
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for Force prediction	
	#building Covariance matrix containers
	K₁ₙₘ= zeros((dim, (1+dim)*num))
	for j in 1:num
		#Fillin convarian of Energy vs Force
		K₁ₙₘ[:,j] = reshape(
					-  kernelfunction(k, X[:,j], xₒ, 1)
				, (dim)
			)
		#Fillin convarian of Force vs Force
		K₁ₙₘ[:, (num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					-  kernelfunction(k, X[:,j], xₒ, 2)
					, (dim, dim)
				)
	end
	return K₁ₙₘ  
end

"""
	Coveriance_fc2(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function 'k') between a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the second order (or harmonic) force constant at the equilibrium point.
"""

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

"""
	Coveriance_fc3(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function 'k') between a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the third order (or cubic anharmonic) force constant at the equilibrium point.
"""

function Coveriance_fc3(X::Matrix{Float64}, xₒ::Vector{Float64}, k)
	dim = size(X,1)
	num = size(X,2)
	
	#building Covariance matrix containers
	K₃ₙₘ= zeros((dim, dim, dim, (1+dim)*num))
	for j in 1:num
		#Fillin convarian of Energy vs FC3
		K₃ₙₘ[:,:,:,j] = reshape(
					-  kernelfunction(k, X[:,j], xₒ, 3)
					, (dim, dim, dim)
				)
		#Fillin convarian of Force vs FC3
		K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					-  kernelfunction(k, X[:,j], xₒ, 4)
					, (dim, dim, dim, dim)
				)
	end
	return K₃ₙₘ
end

"""
    Posterior(Marginal, Covariance, Target)

Calculates a posterior mean. 
The inputs are a marginal likelihood matrix 'Marginal', a covarince variance matrix 'Covarince' 
and a target dataset (including potential energy surfaces and force fields).

SPEEDING UP
Currenly rewritten to do individual tensor contractions with @einsum, as for some reason
directly contracting across two dimensions was ~10x slower. 
"""
function Posterior(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)
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

