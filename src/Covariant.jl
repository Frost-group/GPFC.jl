function Coveriant(X, xₒ, kₒ, σₒ, l, order)
	dim = size(X,1)
	num = size(X,2)
	Kₙₘ= zeros(
		(
			(dim^order), (1+dim)*num
		)
	)
	if order == 0
		for j in 1:num 
			Kₙₘ[j] = 
					kernel(
						kₒ, σₒ, l, xₒ, X[:,j], [order,0]
					)
			Kₙₘ[(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =
					kernel(
						kₒ, σₒ, l, xₒ, X[:,j], [order,1]
					)
		end
	else
		for j in 1:num 
			Kₙₘ[:,j] = 
				reshape(
					kernel(
						kₒ, σₒ, l, xₒ, X[:,j], [order,0]
					),(dim^order, 1)
				)
			Kₙₘ[:,(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] =
				reshape(
					kernel(
						kₒ, σₒ, l, xₒ, X[:,j], [order,1]
					),(dim^order, dim^1)
				)
		end
	end
	return Kₙₘ
end

