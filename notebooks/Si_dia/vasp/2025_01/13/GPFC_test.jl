using Pkg

# Activate Project Environment
Pkg.activate(".")

begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l = 0.30886466
       # Length scale parameter for the kernel

	# Specify the number of training points to be used in the model.
	Num = 100  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
    kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-9  # Energy Gaussian noise
	σₙ = σₑ/l  # Force Gaussian noise 
end;

equi, equi0, x, Target, n = Read_JuLIP_Atoms("notebooks/Si_dia/vasp/Data/d_Si.extxyz",500);
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

println(size(x))

begin
	# Define the kernel scale, which controls the variance of the Gaussian Process.
	σₒ = 0.05  # Kernel Scale (Standard deviation of the Gaussian Process prior)

	# Define the length scale parameter for the kernel, which determines smoothness.
	l1 = 0.4   # Length scale parameter for the kernel
    l2 = 0.30886466
	# Specify the number of training points to be used in the model.
	Num = 100  # Number of training points: Max 509 training points

	# Define the kernel function for the Gaussian Process.
	# It uses a squared exponential kernel with a scaling transformation.
	# σₒ² scales the kernel, and ScaleTransform(l) applies the length scale.
	kernel1 = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l1)
    kernel2 = σₒ^2 * CosineKernel() ∘ ScaleTransform(l2)
    kernel3 = kernel1 + kernel2
    kernel5 = kernel1 * kernel2
    kernel6 = σₒ^2 * Matern52Kernel() ∘ ScaleTransform(l2)
    kernel7 = σₒ^2 * PeriodicKernel() ∘ ScaleTransform(l2)

    # Define Gaussian noise parameters for energy and force measurements.
	σₑ = 1e-9  # Energy Gaussian noise
	σₙ = σₑ/l  # Force Gaussian noise 
end;



X = 0.8 * rand(100) .- 0.4
k1 = [kernel1(0.0, X[ii]) for ii in 1:size(X,1)]
k2 = [kernel2(0.0, X[ii]) for ii in 1:size(X,1)]
k3 = [kernel3(0.0, X[ii]) for ii in 1:size(X,1)]
k4 = [kernel3(0.0, X[ii])^2 for ii in 1:size(X,1)]
k5 = [kernel5(0.0, X[ii]) for ii in 1:size(X,1)]
k6 = [kernel6(0.0, X[ii]) for ii in 1:size(X,1)]
k7 = [kernel7(0.0, X[ii]) for ii in 1:size(X,1)]


kd1 = [kernelfunction1(kernel1, 0.0, X[ii]) for ii in 1:size(X,1)]
kd2 = [kernelfunction2(kernel1, 0.0, X[ii]) for ii in 1:size(X,1)]

kd1_2 = [kernelfunction1(kernel2, 0.0, X[ii]) for ii in 1:size(X,1)]
kd2_2 = [kernelfunction2(kernel2, 0.0, X[ii]) for ii in 1:size(X,1)]

kd1_3 = [kernelfunction1(kernel3, 0.0, X[ii]) for ii in 1:size(X,1)]
kd2_3 = [kernelfunction2(kernel3, 0.0, X[ii]) for ii in 1:size(X,1)]

kd1_7 = [kernelfunction1(kernel7, 0.0, X[ii]) for ii in 1:size(X,1)]
kd2_7 = [kernelfunction2(kernel7, 0.0, X[ii]) for ii in 1:size(X,1)]


scatter(X, k7./maximum(k7))

scatter(X, k6./maximum(k6))
scatter!(X, k1./maximum(k1))

scatter!(X, kd1./maximum(kd1))
scatter!(X, kd2./maximum(kd2))

scatter(X, k2./maximum(k2))
scatter!(X, kd1_2./maximum(kd1_2))
scatter!(X, kd2_2./maximum(kd2_2))

scatter(X, k3./maximum(k3))
scatter!(X, kd1_3./maximum(kd1_3))
scatter!(X, kd2_3./maximum(kd2_3))

scatter(X, k7./maximum(k7))
scatter!(X, kd1_7./maximum(kd1_7))
scatter!(X, kd2_7./maximum(kd2_7))

scatter!(X, k2./maximum(k2))
scatter!(X, k3./maximum(k3))
scatter!(X, k4./maximum(k4))
scatter!(X, k5./maximum(k5))
