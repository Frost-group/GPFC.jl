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
using JuLIP  

include("C://Users//Keerati//Documents//GitHub//GPFC.jl//src//JulibAtoms.jl")
export Read_JuLIP_Atoms
include("C://Users//Keerati//Documents//GitHub//GPFC.jl//src//Analy_kernel.jl")
export rbf_kernel
export kernel_1st_derivative
export kernel_2nd_derivative
export kernel_3nd_derivative
export kernel_4th_derivative
include("C://Users//Keerati//Documents//GitHub//GPFC.jl//src//Posterior.jl")
export rbf_kernel
export Marginal_ana
export Coveriance_fc2_ana
export PosteriorFC2
export run_with_timer
include("C://Users//Keerati//Documents//GitHub//GPFC.jl//src//phRep.jl")
export phonon_dft
export phonon_Γ


begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l = 1.90  # Length scale parameter for the kernel

	# Specify the number of training points to be used in the model.
	Num = 10  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-8  # Energy Gaussian noise
	σₙ = 1e-8  # Force Gaussian noise 
end;

equi, x, Target, e, ∇E,n, mass = Read_JuLIP_Atoms("d_Si.extxyz", Num);

Kₘₘ = @time run_with_timer(Marginal_ana, x, σₑ, σₙ);
K₂ₙₘ = @time run_with_timer(Coveriance_fc2_ana, x, equi);
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

function phonon_data(x::Matrix{Float64}, q::Vector{Float64}, masses::Vector{Float64}, equi::Vector{Float64}, e::Vector{Float64}, ∇E::Vector{Vector{Float64}})
    numAtom = Int(size(masses, 1)/3)
    n = size(x, 2)
    N3 = Int(size(equi, 1)/3/numAtom)

    x_ph = zeros(Float64, (2*numAtom*3, n))
    f_ph = zeros(Float64, (2*numAtom*3, n))

    f = hcat(∇E...)
    R = reshape(equi, 2*3,  N3)

    equi_ph = phonon_dft(equi, q, R, masses)

    for i in 1:n
        x_ph[:,i] = phonon_dft(x[:,i], q, R, masses)
        f_ph[:,i] = phonon_dft(f[:,i], q, R, masses)
    end
    Target_ph = vcat(e, reshape(f_ph, (2*numAtom*3*n,1)))
    return equi_ph, x_ph, f_ph, Target_ph
end


x2d = reshape(x[1:48,2], 2*3, 8)
q = [0., 0., 0.]
R = reshape(equi, 2*3, 8)
m =  vcat(fill(mass[1], 3), fill(mass[2], 3))
f = hcat(∇E...)

begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l = 1.90  # Length scale parameter for the kernel

	# Specify the number of training points to be used in the model.
	Num = 300  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)

    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-8  # Energy Gaussian noise
	σₙ = 1e-8  # Force Gaussian noise 
end;

equi, x, Target, e, ∇E, n, mass = Read_JuLIP_Atoms("d_Si.extxyz", Num);
equi_ph, x_ph, f_ph, Target_ph = phonon_data(x, q, m, equi, e, ∇E);


Kₘₘ_ph = @time run_with_timer(Marginal_ana, x_ph, σₑ, σₙ);
K₂ₙₘ_ph = @time run_with_timer(Coveriance_fc2_ana, x_ph, equi_ph);
FC2_ph = @time run_with_timer(PosteriorFC2, Kₘₘ_ph, K₂ₙₘ_ph, Target_ph);
FC2_ph

heatmap(1:12,
		    1:12, FC2_ph/mass[1],
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="d-Si_FC2 (Traning Data = " *string(n) *")")

sum(FC2_ph[1:6,1:6]/mass[1])
sqrt(mass[1])

FC2_ph[1:6,1:6]/mass[1]
(FC2_ph[1:6,1:6] + FC2_ph[7:12,7:12])/mass[1]

(FC2_ph[1:6,7:12] + FC2_ph[7:12,1:6])/mass[1] 

[ 0.489005 -0.j,  0.03670245+0.j, 0.03670245+0.j, -0.1804861 -0.1804861j, 0.10651578+0.10651578j,  0.10651578+0.10651578j],
[ 0.03670245-0.j,  0.489005  -0.j, 0.03670245+0.j,  0.10651578+0.10651578j, -0.1804861 -0.1804861j ,  0.10651578+0.10651578j],
[ 0.03670245-0.j,  0.03670245-0.j, 0.489005  -0.j,  0.10651578+0.10651578j, 0.10651578+0.10651578j, -0.1804861 -0.1804861j ],
[-0.1804861 +0.1804861j,  0.10651578-0.10651578j, 0.10651578-0.10651578j,  0.489005  -0.j, 0.03670245+0.j,  0.03670245+0.j],
[ 0.10651578-0.10651578j, -0.1804861 +0.1804861j , 0.10651578-0.10651578j,  0.03670245-0.j, 0.489005  -0.j,  0.03670245+0.j],
[ 0.10651578-0.10651578j,  0.10651578-0.10651578j, -0.1804861 +0.1804861j,  0.03670245-0.j, 0.03670245-0.j,  0.489005  -0.j]]