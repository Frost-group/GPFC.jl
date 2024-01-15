### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ f395df88-82eb-11ee-1db9-df6526bca8b5
begin
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum
	using CSV
	using DataFrames
	using DelimitedFiles
	using Plots
end

# ╔═╡ e25d0749-3d13-4fd2-a2c1-938d37e1c01d
begin
	σₒ = 0.05                  # Kernel Scale
	l = 0.4				    # Length Scale
		
	Num = 199                 # Number of training points
	DIM = 3                     # Dimension of Materials
	model = 1                   # Model for Gaussian noise. 1: σₙ = σₑ/l, 2: σₑ =! σₙ 
	order = 1                   # Order of the Answer; 0: Energy, 1: Forces, 2: FC2, 3: FC3
		
	kernel = σₒ^2 * SqExponentialKernel() ∘ ScaleTransform(l)
end;

# ╔═╡ d2f2c14c-841c-4419-aae4-359ace60b830
begin
	σₑ = 1e-9 					# Energy Gaussian noise
	σₙ = 1e-9                   # Force Gaussian noise for Model 2 (σₑ independent)
end

# ╔═╡ f4d7b49e-44a5-4f7a-86c2-37b09d5a9e91
begin
	function kernelfunction1(x₁, x₂)
		return Zygote.gradient( a -> kernel(a, x₂), x₁)[1]
	end
	function kernelfunction2(x₁, x₂)
		return Zygote.hessian(a -> kernel(a, x₂), x₁)
	end
	function kernelfunction3(x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction2(a, x₂), x₁)
	end
	function kernelfunction4(x₁, x₂)
		return ForwardDiff.jacobian(a -> kernelfunction3(a, x₂), x₁)
	end
end

# ╔═╡ 9b59e688-e7ed-4452-a817-90cb29c6700f
function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, numt::Int64, dimA::Int64)
	a  = 4 - dimA
	feature = (CSV.File(FileFeature)|> Tables.matrix)[begin:a:end,2:numt+1]
	
	equi = feature[:,1]
	
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (CSV.File(FileEnergy)|> Tables.matrix)[begin:numt,2]

	force = -reshape((CSV.File(FileForce)|> Tables.matrix)[begin:a:end,2:numt+1], (dim*num,1))

	force[1:dim] = zeros(dim)
		
	Target = vcat(energy, reshape(force, (dim*num,1)))
	
	
	return equi, feature, energy, force, Target
end

# ╔═╡ 9c7f195f-14f0-4457-8573-b48b3c7388a4
function Marginal(X::Matrix{Float64}, σₑ::Float64, σₙ::Float64)
	dim = size(X,1)
	num = size(X,2)
	#building Marginal Likelihood containers
	#For Energy + Force
	KK = zeros(((1+dim)*num, (1+dim)*num))
	#For Energy
	K₀₀ = zeros(((1)*num, (1)*num))
	#For Force
	K₁₁ = zeros(((dim)*num, (dim)*num))
	
	for i in 1:num 
		for j in 1:num 
		#Fillin convarian of Energy vs Energy
			KK[i, j] = kernel(X[:,i], X[:,j])
		#Fillin convarian of Force vs Energy
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernelfunction1(X[:,i], X[:,j])
		#Fillin convarian of Energy vs Force	
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = -KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j]
		#Fillin convarian of Force vs Force
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = -kernelfunction2(X[:,i], X[:,j])
		end
	end

	Iee = σₑ^2 * Matrix(I, num, num)
	Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

	Kₘₘ = KK + II
	
	return Kₘₘ
end

# ╔═╡ dbe46cff-33b3-4322-8853-f08d72fe95f2
function Coveriance_fc3(X::Matrix{Float64}, xₒ::Vector{Float64})
	dim = size(X,1)
	num = size(X,2)
	
	#building Covariance matrix containers
	K₃ₙₘ= zeros((dim, dim, dim, (1+dim)*num))
	for j in 1:num
		#Fillin convarian of Energy vs FC3
		K₃ₙₘ[:,:,:,j] = reshape(
					-  kernelfunction3(X[:,j], xₒ)
					, (dim, dim, dim)
				)
		#Fillin convarian of Force vs FC3
		K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = reshape(
					-  kernelfunction4(X[:,j], xₒ)
					, (dim, dim, dim, dim)
				)
	end
	return K₃ₙₘ
end

# ╔═╡ 1d1e2654-5415-49a4-944f-218d6e3ad31a
function Posterior(Marginal, Covariance, Target)
	dimₚ = size(Covariance, 1)
	dimₜ = size(Marginal, 1)
	Kₘₘ⁻¹ = inv(Marginal)   #
	Kₙₘ = Covariance
	
	MarginalTar = zeros(dimₜ)
	@einsum MarginalTar[m] = Kₘₘ⁻¹[m, n] * Target[n]

	if size(Kₙₘ) == (dimₜ,)
		Meanₚ = Kₙₘ'  * MarginalTar
		
	elseif size(Kₙₘ) == (dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ)
		@einsum Meanₚ[i] = Kₙₘ[i, m] * MarginalTar[m]
	
	elseif size(Kₙₘ) == (dimₚ, dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ, dimₚ)
		@einsum Meanₚ[i, j] = Kₙₘ[i, j, m] * MarginalTar[m]

	elseif size(Kₙₘ) == (dimₚ, dimₚ, dimₚ, dimₜ)
		Meanₚ = zeros(dimₚ, dimₚ, dimₚ)
		@einsum Meanₚ[i, j, k] = Kₙₘ[i, j, k, m] * MarginalTar[m]
	end

	return Meanₚ 
end

# ╔═╡ 053db0a4-70af-43ca-85a4-b4944c1cbb82
begin
	nd = [1,5,10,15,20,30,40,50,60,80,100,130,160,199,250,298]
	P3 = zeros(( 48, 48, 48, size(nd,1)))
	SumRule3 = zeros((size(nd,1)))
end;

# ╔═╡ 957087d5-1e07-4aab-839b-c6427b7f3c53
@time for i in 1:size(nd,1)
	numt1 = nd[i]
	equi, feature, energy, force, Target = ASEFeatureTarget(
    "feature_new", "energy_new", "force_new", numt1, DIM);

	Kₘₘ = Marginal(feature, σₑ, σₙ);
	K₃ₙₘ = Coveriance_fc3(feature, equi);
	Mp = Posterior(Kₘₘ, K₃ₙₘ, Target);
	
	P3[:,:,:,i] = Mp 
	
	SumRule3[i] = abs(sum(Mp))
end 

# ╔═╡ ed69b0c2-d648-4036-9e24-9fd90170057f
animCar = @animate for i in 1:size(nd,1)
	heatmap(1:size(P3[2,:,:,i],1),
		    1:size(P3[2,:,:,i],2), P3[3,:,:,i],
		    c=cgrad(["#064635","#519259", "#96BB7C", "#F0BB62", "#FAD586","#F4EEA9"]),
			aspectratio=:equal,
			size=(700, 700),
		    xlabel="feature coord. (n x d)",
			ylabel="feature coord. (n x d)",
		    title="Si_FC3 (Traning Data = " *string(nd[i]) *")")
end

# ╔═╡ 45a84347-b8c2-4a62-a2b6-4abaff56d198
gif(animCar, "SI2_FC3_1_Cart.gif", fps=2)

# ╔═╡ 2614fa05-7084-4f72-ad74-4add1b71500a
anim = @animate for i in 1:size(nd,1)
	plot(nd[1:i], SumRule3[1:i],
		xlabel="Training points",
		ylabel="Sum of FC3 element",
		xlim = (-1, 305), 
		ylim = (-1.0, 100.0),
		labels = "Cartesian",
		linewidth=3,
		title="Si2_Sum-Rule_FC3 (Traning Data = " *string(nd[i]) *")"
	)
end

# ╔═╡ 23905961-223a-41d5-aa0f-838d4bb7f7d3
gif(anim, "SI2_FC3_Cart.gif", fps=2)

# ╔═╡ 87236719-9144-4654-b0bf-8c07b67ff206
SumRule3

# ╔═╡ 231240be-1616-469a-a7f8-66784ea20642
P3[1,1,1,298]

# ╔═╡ f6f98a96-4259-43dc-9106-e1de09949cf9
let
    x = 1:48
    y = 1:48
    z = 1:48
	
    vol = [FC3[ix,iy,iz] for ix in x, iy in y, iz in z]
    fig, ax, _ = volume(x, y, z, vol, colormap = :plasma,colorrange = (minimum(vol), maximum(vol)),
        figure = (; resolution = (800,800)),  
        axis=(; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,  
            aspect = (1,1,1)))

    fig
end

# ╔═╡ Cell order:
# ╠═f395df88-82eb-11ee-1db9-df6526bca8b5
# ╠═e25d0749-3d13-4fd2-a2c1-938d37e1c01d
# ╠═d2f2c14c-841c-4419-aae4-359ace60b830
# ╠═f4d7b49e-44a5-4f7a-86c2-37b09d5a9e91
# ╠═9b59e688-e7ed-4452-a817-90cb29c6700f
# ╠═9c7f195f-14f0-4457-8573-b48b3c7388a4
# ╠═dbe46cff-33b3-4322-8853-f08d72fe95f2
# ╠═1d1e2654-5415-49a4-944f-218d6e3ad31a
# ╠═053db0a4-70af-43ca-85a4-b4944c1cbb82
# ╠═957087d5-1e07-4aab-839b-c6427b7f3c53
# ╠═ed69b0c2-d648-4036-9e24-9fd90170057f
# ╠═45a84347-b8c2-4a62-a2b6-4abaff56d198
# ╠═2614fa05-7084-4f72-ad74-4add1b71500a
# ╠═23905961-223a-41d5-aa0f-838d4bb7f7d3
# ╠═87236719-9144-4654-b0bf-8c07b67ff206
# ╠═231240be-1616-469a-a7f8-66784ea20642
# ╠═f6f98a96-4259-43dc-9106-e1de09949cf9
