function Coveriant(X, xₒ, k, order)
	dim = size(X,1)
	num = size(X,2)

	Kₙₘ= zeros(
		(
			(dim^order), (1+dim)*num
		)
	)
	if order == 0
		K₀ₙₘ= zeros(
			((1+dim)*num)
		)
		for j in 1:num 
			K₀ₙₘ[j] = 
				kernel(
					k, X[:,j], xₒ, [0,0]
			)
			K₀ₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				kernel(
					k, X[:,j], xₒ, [1,0]
				)
		end
		Kₙₘ = K₀ₙₘ
		
	elseif order == 1
		K₁ₙₘ= zeros(
			(dim, (1+dim)*num)
		)
		for j in 1:num 
			K₁ₙₘ[:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,1]
					), (dim)
				)
			K₁ₙₘ[:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,1]
					), (dim, dim)
				)
		end
		Kₙₘ = K₁ₙₘ  
	
	elseif order == 2
		K₂ₙₘ= zeros(
			(dim, dim, (1+dim)*num)
		)
		for j in 1:num 
			K₂ₙₘ[:,:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,2]
					), (dim, dim)
				)
			K₂ₙₘ[:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,2]
					), (dim, dim, dim)
				)
		end
		Kₙₘ = K₂ₙₘ

	elseif order == 3
		K₃ₙₘ= zeros(
			(dim, dim, dim, (1+dim)*num)
		) 
		for j in 1:num 
			K₃ₙₘ[:,:,:,j] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [0,3]
					), (dim, dim, dim)
				)
			K₃ₙₘ[:,:,:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = 
				reshape(
					kernel(
						k, X[:,j], xₒ, [1,3]
					), (dim, dim, dim, dim)
				)
		end
		Kₙₘ = K₃ₙₘ
	end

	return Kₙₘ
end