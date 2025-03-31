"""
    kernelfunction(k, x₁, x₂::Vector{Float64})

Calculate the derivative of kernel function `k`.
A standard (or compositing) kernel function requires to be defined by using KernelFunctions.jl.
This will turn the defined kernel function and two different atomistic
representation vectors (features) to the derivative of the kernel of those two vectors where the
derivative order can be specified by variable grad. 
"""
function kernelfunction1(kernel, x₁, x₂)
    # Compute the gradient of the kernel with respect to x₁.
    return Zygote.gradient(a -> kernel(a, x₂), x₁)[1]
end
function kernelfunction2(kernel, x₁, x₂)
    # Compute the Hessian (second derivatives) of the kernel with respect to x₁.
    return Zygote.hessian(a -> kernel(a, x₂), x₁)
end
function kernelfunction3(kernel, x₁, x₂)
    # Compute the Jacobian of the Hessian (third derivatives) of the kernel.
    return ForwardDiff.jacobian(a -> kernelfunction2(kernel, a, x₂), x₁)
end
function kernelfunction4(kernel, x₁, x₂)
    # Compute the Jacobian of the third derivatives (fourth derivatives) of the kernel.
    return ForwardDiff.jacobian(a -> kernelfunction3(kernel, a, x₂), x₁)
end

"""
	Marginal(X::Matrix{Float64}, k, l::Float64, σₑ::Float64)

Construct Marginal likelihood matrix of a traning feature `X` (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
through a standard (or compositing) kernel function `k`.
Marginal() function alsom require l (model length scale) and σₑ (scaling factor) used to evalua
ate a model Gaussian noise.
"""
function Marginal(kernel, X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
    # Get dimensions and total size based on input data.
    dim, num = size(X)
    total_size = (1 + dim) * num

    # Preallocate full covariance matrix.
    KK = zeros(total_size, total_size)

    # Fill covariance matrix using the kernel and its derivatives.
    @showprogress "Processing items..." for i in 1:num
        x_i = X[:, i]  # Pre-compute to avoid repeated access
        for j in 1:num
            x_j = X[:, j]

            # Fill covariance components directly.
            KK[i, j] = kernel(x_i, x_j)
            KK[num+1+(i-1)*dim:num+i*dim, j] = kernelfunction1(kernel, x_i, x_j)
            KK[i, num+1+(j-1)*dim:num+j*dim] = -KK[num+1+(i-1)*dim:num+i*dim, j]
            KK[num+1+(i-1)*dim:num+i*dim, num+1+(j-1)*dim:num+j*dim] = -kernelfunction2(kernel, x_i, x_j)
        end
    end

    # Add noise components (preallocated identity matrices).
    Iee = σₑ^2 * I(num)
    Iff = σₙ^2 * I(dim * num)
    Ief = zeros(num, dim * num)
    II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))
    # Compute marginal likelihood.
    Kₘₘ = KK + II
    println("Marginal Likelihood calculated successfully!")
    return Kₘₘ
end


"""
	Coveriance_energy(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function `k`) between a traning feature `X` (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the potential energy surface at the equilibrium point.
"""
function Coveriance_energy(kernel, X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for Energy prediction
	#building Covariance matrix containers
	K₀ₙₘ= zeros(((1+dim)*num))
	@showprogress "Processing items..." for j in 1:num
		vec_j = X[:,j]
		#Fillin convarian of Energy vs Energy
		K₀ₙₘ[j] = kernel(vec_j, xₒ)
		#Fillin convarian of Force vs Energy
		K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =  kernelfunction(kernel, vec_j, xₒ)
	end
	println("Covariance Likelihood calculated successfully!")
	return K₀ₙₘ
end

"""
	Coveriance_force(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function `k`) between a traning feature `X` (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the force fields (which should be zeros) at the equilibrium point.
"""
function Coveriance_force(kernel, X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#Covariance matrix for Force prediction	
	#building Covariance matrix containers
	K₁ₙₘ= zeros((dim, (1+dim)*num))
	@showprogress "Processing items..." for j in 1:num
		vec_j = X[:,j]
		#Fillin convarian of Energy vs Force
		K₁ₙₘ[:,j] = reshape(
					-  kernelfunction1(kernel, vec_j, xₒ)
				, (dim)
			)
		#Fillin convarian of Force vs Force
		K₁ₙₘ[:, (num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					-  kernelfunction2(kernel, vec_j, xₒ)
					, (dim, dim)
				)
	end
	println("Covariance Likelihood calculated successfully!")
	return K₁ₙₘ  
end

"""
	Coveriance_fc2(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function `k`) between a traning feature 'X' (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the second order (or harmonic) force constant at the equilibrium point.
"""
function Coveriance_fc2(kernel, X::Matrix{Float64}, xₒ::Vector{Float64})
    # Get dimensions of input data.
    dim = size(X,1)
    num = size(X,2)

    # Preallocate covariance matrix container.
    K₂ₙₘ = zeros((dim, dim, (1+dim)*num))

    # Process covariance components.
    @showprogress "Processing items..." for j in 1:num
        # Fill covariance of Energy vs FC2.
        K₂ₙₘ[:,:,j] = reshape(
             kernelfunction2(kernel, X[:,j], xₒ)
            , (dim, dim)
        )
        # Fill covariance of Force vs FC2.
        K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
             kernelfunction3(kernel, X[:,j], xₒ)
            , (dim, dim, dim)
        )
    end
    println("Covariance Likelihood calculated successfully!")
    return K₂ₙₘ
end

"""
	Coveriance_fc3(X::Matrix{Float64}, xₒ::Vector{Float64}, k)

Contruct Covariance matrix (based on a kernel function `k`) between a traning feature `X`` (an atomistic
representation vector of target potential energy surfaces and corresponding force fields) 
and xₒ, the atomistic equilibrium structure vector.

This will be used to evaluate the third order (or cubic anharmonic) force constant at the equilibrium point.
"""
function Coveriance_fc3(kernel, X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#building Covariance matrix containers
	K₃ₙₘ= zeros((dim, dim, dim, (1+dim)*num))
	@showprogress "Processing items..." for j in 1:num
		vec_j = X[:,j]
		#Fillin convarian of Energy vs FC3
		K₃ₙₘ[:,:,:,j] = reshape(
					-  kernelfunction3(kernel, vec_j, xₒ)
					, (dim, dim, dim)
				)
		#Fillin convarian of Force vs FC3
		K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					-  kernelfunction4(kernel, vec_j, xₒ)
					, (dim, dim, dim, dim)
				)
	end
	return K₃ₙₘ
end

"""
    Posterior(Marginal, Covariance, Target)

Calculates a posterior mean. 
The inputs are a marginal likelihood matrix `Marginal`, a covarince variance matrix `Covarince` 
and a target dataset (including potential energy surfaces and force fields).

SPEEDING UP
Currenly rewritten to do individual tensor contractions with @einsum, as for some reason
directly contracting across two dimensions was ~10x slower. 
"""
function Posterior_energy(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)  
	Kₙₘ = Covariance
	
	MarginalTar = zeros(dimₜ)
	@einsum MarginalTar[m] = Kₘₘ⁻¹[m, n] * Target[n]
	
	size(Kₙₘ) == (dimₜ,)
	Meanₚ = Kₙₘ'  * MarginalTar

	println("energy calculated successfully!")
	return Meanₚ 
end

function Posterior_force(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)  
	Kₙₘ = Covariance
	
	MarginalTar = zeros(dimₜ)
	@einsum MarginalTar[m] = Kₘₘ⁻¹[m, n] * Target[n]
	
	size(Kₙₘ) == (dimₚ, dimₜ)
	Meanₚ = zeros(dimₚ)
	@einsum Meanₚ[i] = Kₙₘ[i, m] * MarginalTar[m]

	println("force calculated successfully!")
	return Meanₚ 
end

function PosteriorFC2(Marginal, Covariance, Target)
    # Get dimensions for processing.
    dimₚ, _, dimₜ = size(Covariance)

    # Use efficient linear algebra to solve systems instead of matrix inversion.
    Kₘₘ⁻¹_Target = Marginal \ Target  # Solves Marginal * x = Target

    # Preallocate result matrix.
    Meanₚ = zeros(dimₚ, dimₚ)

    # Compute Meanₚ using einsum for efficiency.
    @einsum Meanₚ[i, j] = Covariance[i, j, m] * Kₘₘ⁻¹_Target[m]

    println("FC2 calculated successfully!")
    return Meanₚ 
end

function PosteriorFC3(Marginal, Covariance, Target)
	# Get dimensions for processing.
    dimₚ, _, dimₜ = size(Covariance)

    # Use efficient linear algebra to solve systems instead of matrix inversion.
    Kₘₘ⁻¹_Target = Marginal \ Target  # Solves Marginal * x = Target

	# Preallocate result matrix.
	Meanₚ = zeros(dimₚ, dimₚ, dimₚ)

	@einsum Meanₚ[i, j, k] =  Covariance[i, j, k, m] * Kₘₘ⁻¹_Target[m]

	println("FC3 calculated successfully!")
	return Meanₚ 
end

function rbf_kernel(x::Vector{Float64}, x_star::Vector{Float64}, σ::Float64, l::Float64)
    r = x .- x_star
    d2 = sum(r .^ 2)
    return σ * exp(-0.5 * d2 / l^2)
end

function Marginal_ana(X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
    # Get dimensions and total size based on input data.
    dim, num = size(X)
    total_size = (1 + dim) * num

    # Preallocate full covariance matrix.
    KK = zeros(total_size, total_size)

    # Fill covariance matrix using the kernel and its derivatives.
    @showprogress "Processing items..." for i in 1:num
        x_i = X[:, i]  # Pre-compute to avoid repeated access
        for j in 1:num
            x_j = X[:, j]

            # Fill covariance components directly.
            KK[i, j] = rbf_kernel(x_i, x_j, σₒ, l)
            KK[num+1+(i-1)*dim:num+i*dim, j] = kernel_1st_derivative(x_i, x_j, σₒ, l)
            KK[i, num+1+(j-1)*dim:num+j*dim] = -KK[num+1+(i-1)*dim:num+i*dim, j]
            KK[num+1+(i-1)*dim:num+i*dim, num+1+(j-1)*dim:num+j*dim] = -kernel_2nd_derivative(x_i, x_j, σₒ, l)
        end
    end

    # Add noise components (preallocated identity matrices).
    Iee = σₑ^2 * I(num)
    Iff = σₙ^2 * I(dim * num)
    Ief = zeros(num, dim * num)

    II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

    # Compute marginal likelihood.
    Kₘₘ = KK + II
    println("Marginal Likelihood calculated successfully!")
    return Kₘₘ
end

function Coveriance_fc2_ana(X::Matrix{Float64}, xₒ::Vector{Float64})
    # Get dimensions of input data.
    dim = size(X,1)
    num = size(X,2)

    # Preallocate covariance matrix container.
    K₂ₙₘ = zeros((dim, dim, (1+dim)*num))

    # Process covariance components.
    @showprogress "Processing items..." for j in 1:num
        # Fill covariance of Energy vs FC2.
        K₂ₙₘ[:,:,j] = reshape(
            kernel_2nd_derivative(X[:,j], xₒ, σₒ, l)
            , (dim, dim)
        )
        # Fill covariance of Force vs FC2.
        K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
            kernel_3rd_derivative(X[:,j], xₒ, σₒ, l)
            , (dim, dim, dim)
        )
    end
    println("Covariance Likelihood calculated successfully!")
    return K₂ₙₘ
end

function run_with_timer(task_function, args...)
    # Print a starting message.
    println("Starting task...")

    # Measure elapsed time during task execution.
    elapsed_time = @elapsed results = task_function(args...)

    # Print the elapsed time.
    println("Calculation time: $(elapsed_time) seconds")
    return results 
end