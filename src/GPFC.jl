module GPFC

# Import Packages

# Core Libraries
#using LinearAlgebra      # Linear algebra operations
#using StaticArrays
#using Random
#using Statistics         # Basic statistics
#using DelimitedFiles     # Work with delimited files

# Data Handling and I/O
#using CSV                # Read/write CSV files
#using DataFrames         # Tabular data manipulation

# Automatic Differentiation
#using ForwardDiff        # Forward-mode AD
#using Zygote             # Reverse-mode AD

# Kernel Methods and Tensor Operations
#using KernelFunctions    # Kernel methods for ML
#using Einsum             # Einstein summation notation

# Visualization and Utilities
#using Plots              # Visualization
#using StatsBase          # Extended statistics functions
#using ProgressMeter      # Progress bar display

# Atomistic Simulations
#using JuLIP              # Atomistic simulations
#using Quaternions

include("JulibAtoms.jl")
export Read_JuLIP_Atoms 
export quaternion_to_rotation_matrix
export rotate_3n_points
export Read_JuLIP_Atoms_rotation

include("csv.jl")
export ASEFeatureTarget

include("Posterior.jl")
export kernelfunction1
export kernelfunction2
export kernelfunction3
export kernelfunction4
export Marginal
export Coveriance_energy
export Coveriance_force
export Coveriance_fc2
export Coveriance_fc3
export Posterior_energy
export Posterior_force
export PosteriorFC2
export PosteriorFC3

include("Analy_kernel.jl")
export rbf_kernel
export kernel_1st_derivative
export kernel_2nd_derivative
export kernel_3nd_derivative
export kernel_4th_derivative

end

