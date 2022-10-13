function Coveriant(X, Xₒ, kₒ, σₒ, l, order)
	dim = size(X,1)
	num_train = size(X,2)
	num_test = size(Xₒ,2)

	Kₙₘ= zeros(
		(
			(dim^order)*num_test, (1+dim)*num_train
		)
	)
	if order == 0
		for i in 1:num_test 
			for j in 1:num_train
				Kₙₘ[i,j] = 
						kernel(
							kₒ, σₒ, l, Xₒ[:,i], X[:,j], [order,0]
						)
		
				Kₙₘ[i,(num_train+1)+((j-1)*dim):(num_train+1)+((j)*dim)-1] =
						kernel(
							kₒ, σₒ, l, Xₒ[:,i], X[:,j], [order,1]
						)
			end
		end
	else
		for i in 1:num_test 
			for j in 1:num_train
				Kₙₘ[1+((i-1)*dim^order):((i)*dim^order),j] = 
					reshape(
						kernel(
							kₒ, σₒ, l, Xₒ[:,i], X[:,j], [order,0]
						),(dim^order, 1)
					)
		
				Kₙₘ[1+((i-1)*dim^order):((i)*dim^order),(num_train+1)+((j-1)*dim):(num_train+1)+((j)*dim)-1] =
					reshape(
						kernel(
							kₒ, σₒ, l, Xₒ[:,i], X[:,j], [order,1]
						),(dim^order, dim^1)
					)
			end
		end
	end

	return Kₙₘ
end