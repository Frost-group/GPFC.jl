function Posterior(X, xₒ, Target, kₒ, σₒ, l, order)
	dim = size(X,1)
	Kₘₘ⁻¹ = Marginal(X, kₒ, σₒ, l)
	Kₙₘ = Coveriant(X, xₒ, kₒ, σₒ, l, order)
	Meanₚ = Kₙₘ * Kₘₘ⁻¹ * Target	
	return Meanₚ
end