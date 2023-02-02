function Marginal(X, k, σₑ, σₙ; model = 1)
	dim = size(X,1)
	num = size(X,2)
	KK = zeros(
		(
			(1+dim)*num, (1+dim)*num
		)
	)
	K₀₀ = zeros(
		(
			(1)*num, (1)*num
		)
	)
	K₁₁ = zeros(
		(
			(dim)*num, (dim)*num
		)
	)
	
	for i in 1:num 
		for j in 1:num 
			KK[i, j] = kernel(
				k, X[:,i], X[:,j], [0,0]
			)
			
			KK[(num+1)+((i-1)*dim): (num+1)+((i)*dim)-1,j] = kernel(
				k, X[:,i], X[:,j], [1,0]
			)
			
			KK[i,(num+1)+((j-1)*dim): (num+1)+((j)*dim)-1] = kernel(
				k, X[:,i], X[:,j], [0,1]
			)
			
			KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1,
				(num+1)+((j-1)*dim):(num+1)+((j)*dim)-1] = kernel(
					k, X[:,i], X[:,j], [1,1]
				)
			
			K₀₀[i, j] = KK[i, j]
			K₁₁[(1)+((i-1)*dim):(1)+((i)*dim)-1,
				(1)+((j-1)*dim):(1)+((j)*dim)-1] = KK[(num+1)+((i-1)*dim):(num+1)+((i)*dim)-1, (num+1)+((j-1)*dim):(num+1)+((j)*dim)-1]
			
		end
	end
	
	if model == 1
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = (σₑ / l)^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
	else
		Iee = σₑ^2 * Matrix(I, num, num)
		Iff = σₙ^2 * Matrix(I, dim * num, dim * num)
		Ief = zeros(num, dim * num)
		II = vcat(hcat(Iee, Ief), hcat(Ief', Iff))

		Kₘₘ = KK + II
		K₀₀ = K₀₀ + Iee 
		K₁₁ = K₁₁ + Iff
	end
	#Kₘₘ⁻¹ = inv(KK+II)
	return K₀₀, K₁₁, Kₘₘ
end