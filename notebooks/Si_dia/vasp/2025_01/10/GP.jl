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
	σₑ = 1e-8  # Energy Gaussian noise
	σₙ = 1e-8  # Force Gaussian noise 
end;

# Import packages.
using AbstractGPs, Plots

# Generate toy synthetic data.
x = rand(10,2)
y = sin.(x)

# Define GP prior with Matern-3/2 kernel
f = GP(Matern32Kernel())

# Finite projection of `f` at inputs `x`.
# Added Gaussian noise with variance 0.001.
fx = f(x, 0.001)

# Log marginal probability of `y` under `f` at `x`.
# Quantity typically maximised to train hyperparameters.
logpdf(fx, y)

# Exact posterior given `y`. This is another GP.
p_fx = posterior(fx, y)

# Log marginal posterior predictive probability.
logpdf(p_fx(x), y)

# Plot posterior.
scatter(x, y; label="Data")
plot!(-0.5:0.001:1.5, p_fx; label="Posterior")