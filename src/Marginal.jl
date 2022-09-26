
function Marginal(X, kₒ, σₒ, l)
	dim = size(X,1)
	num = size(X,2)
	KK = zeros(
		(
			(1+dim)*num, (1+dim)*num
		)
	)

	for i in 1:num 
		for j in 1:num 
			KK[i, j] = kernel(
				kₒ, σₒ, l, X[:,i], X[:,j], [0,0]
			)
			
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernel(
				kₒ, σₒ, l, X[:,i], X[:,j], [1,0]
			)
			
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = kernel(
				kₒ, σₒ, l, X[:,i], X[:,j], [0,1]
			)
			
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,
				(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = kernel(
					kₒ, σₒ, l, X[:,i], X[:,j], [1,1]
				)
		end
	end
	Iee = σₒ * 10e-3 * Matrix(I, num, num)
	Iff = σₒ * 10e-3 / l * Matrix(I, dim * num, dim * num)
	Ief = zeros(num, dim * num)
	II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))
	Kₘₘ⁻¹ = inv(KK+II)
	return Kₘₘ⁻¹
end