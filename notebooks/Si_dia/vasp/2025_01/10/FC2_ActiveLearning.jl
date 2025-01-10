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
    "Random",
    "AbstractGPs",
    "Stheno",
    "Optim"
])

using KernelFunctions
using AbstractGPs
using Stheno
using LinearAlgebra
using Random

# Define a 3N-dimensional feature space (for simplicity, assume N=2)
N = 2  # Number of atoms
D = 3 * N  # Dimensionality of features

# Simulate some initial training data
Random.seed!(42)
n_train = 5  # Initial training points
n_pool = 20  # Pool of unlabeled data points

X_train = randn(n_train, D)  # Initial training data
Y_train = sin.(sum(X_train, dims=2)) + 0.1 * randn(n_train)  # Initial training labels

X_pool = randn(n_pool, D)  # Unlabeled pool data

σₒ = 0.05
l = 0.4

function build_gp(X, Y)
    # Use Stheno.SE kernel
    kernel = (σₒ^2) * SqExponentialKernel() ∘ ScaleTransform(l)  # Scale = 1.0, Length scale = 0.5
    f = GP(kernel)

    # Observations as a FiniteGP
    obs = f(X', 0.1)  # Use noise variance = 0.1 (Ensure X' is transposed correctly)
    
    # Compute posterior
    return posterior(obs, Y)
end


# Active learning loop
n_iterations = 10  # Number of active learning iterations

for iter in 1:n_iterations
    println("Iteration $iter")

    # Build GP model with current training data
    post = build_gp(X_train, Y_train)

    # Predict uncertainty on the pool
    uncertainties = map(x -> var(post.f(x)), eachrow(X_pool))

    # Select the point with maximum uncertainty
    max_uncertainty_idx = argmax(uncertainties)
    next_point = X_pool[max_uncertainty_idx, :]

    # Simulate a new observation (replace with your actual data acquisition)
    next_label = sin(sum(next_point)) + 0.1 * randn()

    # Update training data
    X_train = vcat(X_train, next_point')
    Y_train = vcat(Y_train, next_label)

    # Remove the selected point from the pool
    X_pool = vcat(X_pool[1:max_uncertainty_idx-1, :], X_pool[max_uncertainty_idx+1:end, :])

    # Print progress
    println("Selected next point with uncertainty: $(uncertainties[max_uncertainty_idx])")
end

println("Final training data size: $(size(X_train))")
