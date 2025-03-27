using Pkg

# Activate Project Environment
Pkg.activate(".")
Base.active_project()
Pkg.instantiate()
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


function Read_JuLIP_Atoms(extxyz_filename::String, num_structure)
    # Read data from the specified extended XYZ file.
    data = read_extxyz(extxyz_filename)

    # Number of structures to process.
    n = num_structure

    # Extract atomic positions for each structure and flatten them.
    X = [reduce(vcat, data[ii].X) for ii in 1:n]

    # Store the equilibrium structure (positions of the first structure).
    equi = X[1]

    # Horizontally Concatenate all atomic positions into a matrix.
    x = hcat(X...)

    # Extract energy data for each structure.
    E = [get_data(data[ii], "energy") for ii in 1:n]

    # Combine all energies into a single vector.
    e = vcat(E...)

    # Extract and flatten forces for each structure (negative gradient of energy).
    ∇E = [-reduce(vcat, get_data(data[ii], "forces")) for ii in 1:n]

    # Vertically concatenate all forces into a single vector.
    ∇e = vcat(∇E...)

    # Combine energies and forces into the target output.
    Target = vcat(e, ∇e)

    # Return the equilibrium structure, positions, target values, and structure count.
    return equi, x, Target, n
end


 # RBF Kernel
function rbf_kernel(x::Vector{Float64}, x_star::Vector{Float64}, σ::Float64, l::Float64)
    r = x .- x_star
    d2 = sum(r .^ 2)
    return σ^2 * exp(-0.5 * d2 / l^2)
end

# 1st-Order Derivative (Gradient)
function kernel_1st_derivative(x, x_star, σ, l)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    return -k_val * r / l^2
end

# 2nd-Order Derivative (Hessian)
function kernel_2nd_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    H = zeros(n, n)
    for i in 1:n
        for j in 1:n
            delta = i == j ? 1.0 : 0.0
            H[i, j] = k_val * (r[i] * r[j] / l^4 - delta / l^2)
        end
    end
    return H
end

# 3rd-Order Derivative (Rank-3 Tensor)
function kernel_3rd_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    T = zeros(n, n, n)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                T[i, j, k] = k_val * (
                    -r[i] * r[j] * r[k] / l^6 +
                    ((i == j ? r[k] : 0.0) +
                     (i == k ? r[j] : 0.0) +
                     (j == k ? r[i] : 0.0)) / l^4
                )
            end
        end
    end
    return T
end

# 4th-Order Derivative (Rank-4 Tensor)
function kernel_4th_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    T4 = zeros(n, n, n, n)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                for m in 1:n
                    term1 = r[i] * r[j] * r[k] * r[m] / l^8
                    term2 = (
                        (i == j ? r[k]*r[m] : 0.0) +
                        (i == k ? r[j]*r[m] : 0.0) +
                        (i == m ? r[j]*r[k] : 0.0) +
                        (j == k ? r[i]*r[m] : 0.0) +
                        (j == m ? r[i]*r[k] : 0.0) +
                        (k == m ? r[i]*r[j] : 0.0)
                    ) / l^6
                    term3 = (
                        (i == j && k == m ? 1.0 : 0.0) +
                        (i == k && j == m ? 1.0 : 0.0) +
                        (i == m && j == k ? 1.0 : 0.0)
                    ) / l^4
                    T4[i, j, k, m] = k_val * (term1 - term2 + term3)
                end
            end
        end
    end
    return T4
end

function Marginal(X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
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

function Coveriance_fc2(X::Matrix{Float64}, xₒ::Vector{Float64})
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

function run_with_timer(task_function, args...)
    # Print a starting message.
    println("Starting task...")

    # Measure elapsed time during task execution.
    elapsed_time = @elapsed results = task_function(args...)

    # Print the elapsed time.
    println("Calculation time: $(elapsed_time) seconds")
    return results 
end


begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l = 1.90  # Length scale parameter for the kernel

	# Specify the number of training points to be used in the model.
	Num = 500  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-9  # Energy Gaussian noise
	σₙ = 1e-9  # Force Gaussian noise 
end;

equi, x, Target, n = Read_JuLIP_Atoms("d_Si.extxyz", Num);

Kₘₘ = @time run_with_timer(Marginal, x, σₑ, σₙ);
K₂ₙₘ = @time run_with_timer(Coveriance_fc2, x, equi);
FC2 = @time run_with_timer(PosteriorFC2, Kₘₘ, K₂ₙₘ, Target);

FC2

heatmap(1:48,
		    1:48, FC2,
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n) *")")

sum(FC2)