function Posterior(X, xₒ, Target, k, σₙ, order)
	dim = size(X,1)
	num = size(X,2)
	Kₘₘ⁻¹ = invMarginal(X, k, σₙ)
	Kₙₘ = Coveriant(X, xₒ, k, order)
	
	if order == 0
		Meanₚ = Kₙₘ' * Kₘₘ⁻¹ * Target
		
	elseif order == 1
		Meanₚ = ones(dim)
		@einsum Meanₚ[i] = Kₙₘ[i, m] * Kₘₘ⁻¹[m, n] * Target[n]
	
	elseif order == 2
		Meanₚ = ones(dim, dim)
		@einsum Meanₚ[i, j] = Kₙₘ[i, j, m] * Kₘₘ⁻¹[m, n] * Target[n]

	elseif order == 3
		Meanₚ = ones(dim, dim, dim)
		@einsum Meanₚ[i, j, k] = Kₙₘ[i, j, k, m] * Kₘₘ⁻¹[m, n] * Target[n]
	end
	
	return Meanₚ
end