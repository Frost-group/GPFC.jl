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
    "JuLIP"              # Atomistic simulations
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

data = read_extxyz("/Users/paintokk/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2024_12/12/d_Si.extxyz")

