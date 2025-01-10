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
    "Quaternions",
    "StaticArrays",
    "Random"  
])

# Import Packages

# Core Libraries
using LinearAlgebra      # Linear algebra operations
using StaticArrays
using Random
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
using Quaternions

begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l = 0.4    # Length scale parameter for the kernel

	# Specify the number of training points to be used in the model.
	Num = 100  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-5  # Energy Gaussian noise
	σₙ = 1e-5  # Force Gaussian noise 
end;

# Function to read and process atomic structures from an extended XYZ file.
function Read_JuLIP_Atoms(extxyz_filename::String, num_structure)
    # Read data from the specified extended XYZ file.
    data = read_extxyz(extxyz_filename)

    # Number of structures to process.
    n = num_structure

    # Extract atomic positions for each structure and flatten them.
    X1 = [reduce(vcat, data[ii].X) for ii in 1:n]
    X0 = [reduce(vcat, data[1].X) for _ in 1:n]
    X = X1 - X0 

    # Store the equilibrium structure (positions of the first structure).
    equi = X[1]
    equi0 = X1[1]
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
    return equi, equi0, x, Target, n
end

# Generate a normalized Quaternion with normally distributed random 4 component
Random.seed!(2)
qr = normalize(Quaternion(randn(),randn(),randn(),randn()));

"""
    quaternion_to_rotation_matrix(q::Quaternion)

Convert a (unit) Quaternion `q` into the corresponding 3×3 rotation matrix.
"""
function quaternion_to_rotation_matrix(q::Quaternion)
    # Make sure q is normalized (if not already). 
    # If `q` is already a unit quaternion, you can skip this step or check norm ~ 1.0.

    # Extract components
    w = real(q)                 # scalar part
    (x, y, z) = imag_part(qr)   # i, j ,k component

    # For a unit quaternion q = w + xi + yj + zk
    # the rotation matrix R is given by (in row-major form):
    #
    # R =  [ w^2 + x^2 - y^2 - z^2    2xy - 2wz           2xz + 2wy
    #        2xy + 2wz              w^2 - x^2 + y^2 - z^2 2yz - 2wx
    #        2xz - 2wy              2yz + 2wx            w^2 - x^2 - y^2 + z^2 ]
    #
    # This can be derived from q * v * q⁻¹ for a vector v.

    return @SMatrix [
        w^2 + x^2 - y^2 - z^2  2x*y - 2w*z            2x*z + 2w*y
        2x*y + 2w*z            w^2 - x^2 + y^2 - z^2  2y*z - 2w*x
        2x*z - 2w*y            2y*z + 2w*x            w^2 - x^2 - y^2 + z^2
    ]
end

# This should be an evenly distributed rotation matrix
rotate1 = quaternion_to_rotation_matrix(qr)

function rotate_3n_points(R::AbstractMatrix{<:Real}, coords::AbstractVector{<:Real})
    # Ensure coords has length multiple of 3
    @assert length(coords) % 3 == 0 "coords must have length multiple of 3"
    N = length(coords) ÷ 3
    
    # Reshape coords to a 3 x N matrix 
    #   M[:, i] = ( x_i, y_i, z_i )
    M = reshape(coords, 3, N)
    
    # Apply the rotation matrix: R * M  =>  3 x N
    M_rot = R * M
    
    # Flatten back into 3N-vector
    return vec(M_rot)
end

function Read_JuLIP_Atoms_rotation(extxyz_filename::String, num_structure, num_rotation)
    data = read_extxyz(extxyz_filename)

    n = num_structure

    # Extract atomic positions for each structure and flatten them
    X11 = [reduce(vcat, data[ii].X) for ii in 1:n]
    X0 = [reduce(vcat, data[1].X) for _ in 1:n]
    X1 = X11 .- X0  # Subtract reference positions
    equi = X1[1]
    # Extract energies and forces for each structure
    E1 = [get_data(data[ii], "energy") for ii in 1:n]
    ∇E1 = [-reduce(vcat, get_data(data[ii], "forces")) for ii in 1:n]

    # Initialize storage vectors
    X = [X1[1]]  # Include the first structure
    E = [E1[1]]
    ∇E = [∇E1[1]]

# Loop over structures and rotations
    for ii in 2:n
        # Add the original structure data
        push!(X, X1[ii])
        push!(E, E1[ii])
        push!(∇E, ∇E1[ii])

        # Generate rotations
        for jj in 1:num_rotation
            Random.seed!(jj)  # Seed random number generator

            # Generate a random quaternion and normalize it
            qr = normalize(Quaternion(randn(), randn(), randn(), randn()))

            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qr)

            # Rotate positions and forces, append results
            push!(X, rotate_3n_points(R, X1[ii]))
            push!(E, E1[ii])  # Energy remains unchanged for rotations
            push!(∇E, rotate_3n_points(R, ∇E1[ii]))
        end
    end

    # Concatenate final results
    X_final = hcat(X...)  # Combine all vectors in X into a matrix
    E_final = E  # Energies are stored as a vector
    ∇E_final = vcat(∇E...)  # Combine all force vectors row-wise
    Target_final = vcat(E_final, ∇E_final)  # Combine energy and force data


    # Return the equilibrium structure, positions, target values, and structure count.
    return equi, X_final, Target_final, n, num_rotation
end


# Read and process atomic structures from a specified file.
# Input: Extended XYZ file path and number of structures to process.
# Output: Equilibrium positions, concatenated atomic positions, target values, and structure count.
equi, equi0, x, Target, n = Read_JuLIP_Atoms("notebooks/Si_dia/vasp/Data/d_Si.extxyz", 50);
equi1, x1, Target1, n1, nr1 = Read_JuLIP_Atoms_rotation("notebooks/Si_dia/vasp/Data/d_Si.extxyz", 50, 5);
# Define functions to compute derivatives and Hessians of the kernel.

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

# Function to compute the marginal likelihood for Gaussian Processes.
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

# Function to compute covariance for FC2 prediction.
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

# Function to compute the posterior mean for FC2.
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

# Function to time the execution of a task function.
function run_with_timer(task_function, args...)
    # Print a starting message.
    println("Starting task...")

    # Measure elapsed time during task execution.
    elapsed_time = @elapsed results = task_function(args...)

    # Print the elapsed time.
    println("Calculation time: $(elapsed_time) seconds")
    return results 
end

# Compute covariance matrices and posterior estimates with timing.
Kₘₘ = @time run_with_timer(Marginal, kernel, x, σₑ, σₙ);
K₂ₙₘ = @time run_with_timer(Coveriance_fc2, kernel, x, equi);
FC2 = @time run_with_timer(PosteriorFC2, Kₘₘ, K₂ₙₘ, Target);

Kₘₘ1 = @time run_with_timer(Marginal, kernel, x1, σₑ, σₙ);
K₂ₙₘ1 = @time run_with_timer(Coveriance_fc2, kernel, x1, equi1);
FC21 = @time run_with_timer(PosteriorFC2, Kₘₘ1, K₂ₙₘ1, Target1);


x1

Kₘₘ

inv(Kₘₘ1) 

FC2 

FC21


heatmap(1:14455,
		    1:14455, inv(Kₘₘ1),
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n) *")")


heatmap(1:48,
		    1:48, FC2,
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n) *")")

heatmap(1:48,
		    1:48, FC21,
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n1) *")")

