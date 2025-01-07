using Pkg

# Activate Project Environment
Pkg.activate(".")

# Add Required Packages
Pkg.add([
    "KernelFunctions",   # Kernel methods for ML
    "ForwardDiff",       # Automatic differentiation
    "Zygote",            # Gradient-based AD
    "LinearAlgebra",     # Linear algebra operations
    "Einsum",            # Einstein summation
    "Statistics",        # Statistical computations
    "Plots",             # Visualization library
    "StatsBase",         # Basic statistics
    "CSV",               # Read/write CSV files
    "DataFrames",        # DataFrame for structured data
    "DelimitedFiles",    # Work with delimited text files
    "ProgressMeter",     # Display progress bars
    "JuLIP",             # Atomistic simulations
    "StaticArraysCore",
    "Optim"
])

# Import Packages

# Core Libraries
using LinearAlgebra      # Linear algebra operations
using Statistics         # Basic statistics
using DelimitedFiles     # Work with delimited files

# Data Handling and I/O
using CSV                # Read/write CSV files
using DataFrames         # Tabular data manipulation

# Automatic Differentiation
using ForwardDiff        # Forward-mode AD
using Zygote             # Reverse-mode AD

# Kernel Methods and Tensor Operations
using KernelFunctions    # Kernel methods for ML
using Einsum             # Einstein summation notation

# Visualization and Utilities
using Plots              # Visualization
using StatsBase          # Extended statistics functions
using ProgressMeter      # Progress bar display

# Atomistic Simulations
using JuLIP              # Atomistic simulations

using StaticArraysCore

using Optim

#data = read_extxyz("/Users/paintokk/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2024_12/12/d_Si.extxyz")

function Read_JuLIP_Atoms(extxyz_filename::String, num_structure)
    data = read_extxyz(extxyz_filename)

    n = num_structure
    X = [reduce(vcat, data[ii].X) for ii in 1:n]
    equi = X[1]
    x = hcat(X...)
    E = [get_data(data[ii], "energy") for ii in 1:n]
    e = vcat(E...)
    ∇E = [-reduce(vcat, get_data(data[ii], "forces")) for ii in 1:n]
    ∇e = vcat(∇E...)
    Target = vcat(e, ∇e)
    
    return equi, x, Target, n
end

equi, x, Target, n = Read_JuLIP_Atoms("notebooks/Si_dia/vasp/2024_12/12/d_Si.extxyz", Num);

begin
	σₒ = 0.05                   # Kernel Scale
	l = 0.4		    
	Num = 100                  # Number of training points
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

function Marginal(kernel, X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
    dim, num = size(X)
    total_size = (1 + dim) * num
    
    # Preallocate full covariance matrix
    KK = zeros(total_size, total_size)
    
    @showprogress "Processing items..." for i in 1:num
        x_i = X[:, i]  # Pre-compute to avoid repeated access
        for j in 1:num
            x_j = X[:, j]

            # Fill covariance components directly
            K_ij = kernel(x_i, x_j)
            KK[i, j] = K_ij
            KK[num+1+(i-1)*dim:num+i*dim, j] = kernelfunction1(kernel, x_i, x_j)
            KK[i, num+1+(j-1)*dim:num+j*dim] = -KK[num+1+(i-1)*dim:num+i*dim, j]
            KK[num+1+(i-1)*dim:num+i*dim, num+1+(j-1)*dim:num+j*dim] = -kernelfunction2(kernel, x_i, x_j)
        end
    end
    
    # Add noise components (preallocated identity matrices)
    Iee = σₑ^2 * I(num)
    Iff = σₙ^2 * I(dim * num)
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
    dimₚ, _, dimₜ = size(Covariance)

    # Use more efficient linear algebra for solving systems instead of matrix inversion
    Kₘₘ⁻¹_Target = Marginal \ Target  # Solves Marginal * x = Target

    # Preallocate result matrix
    Meanₚ = zeros(dimₚ, dimₚ)

    # Compute Meanₚ using einsum for efficiency
    @einsum Meanₚ[i, j] = Covariance[i, j, m] * Kₘₘ⁻¹_Target[m]

    println("FC2 calculated successfully!")
    return Meanₚ 
end

function run_with_timer(task_function, args...)
    println("Starting task...")
    elapsed_time = @elapsed results = task_function(args...)
    println("Calculation time: $(elapsed_time) seconds")
	return results 
end

Kₘₘ = @time run_with_timer(Marginal, kernel, x, σₑ, σₙ);
K₂ₙₘ = @time run_with_timer(Coveriance_fc2, kernel, x, equi);
FC2 = @time run_with_timer(PosteriorFC2, Kₘₘ, K₂ₙₘ, Target);

heatmap(1:48,
		    1:48, FC2,
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n) *")")


function negative_log_marginal_likelihood(params, X, Target)
    # Extract hyperparameters
    σₒ, l, σₑ, σₙ = params
            
    # Define kernel with updated parameters
    kernel = (σₒ^2) * SqExponentialKernel() ∘ ScaleTransform(l)
            
    # Compute Marginal Likelihood
    Kₘₘ = Marginal(kernel, X, σₑ, σₙ)

    ϵ = 1e-8  # Jitter for numerical stability
    Kₘₘ += ϵ * I(size(Kₘₘ, 1))  # Add small value to diagonal
    Kₘₘ = (Kₘₘ + Kₘₘ') / 2

        # Compute log determinant and quadratic term
    L = cholesky(Kₘₘ)    # Use Cholesky decomposition for stability
    logdetK = 2 * sum(log.(diag(L.U)))  # Log determinant
    quad_term = Target' * (L \ (L' \ Target))  # Quadratic term using linear solver
        
    # Negative log marginal likelihood
    return 0.5 * (quad_term + logdetK + length(Target) * log(2π))
end

# Initial guess for hyperparameters
initial_params = [0.05, 0.4, 1e-5, 1e-5]

# Optimize using Nelder-Mead or BFGS
result = optimize(
    params -> negative_log_marginal_likelihood(params, x, Target),  # Loss function
    initial_params,                                                 # Initial guess
    NelderMead()                                                    # Optimization algorithm
)