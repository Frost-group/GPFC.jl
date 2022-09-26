
module GPFC


begin
	using KernelFunctions
	using ForwardDiff
	using Plots
	using LinearAlgebra
end

function kernel(kₒ, σₒ , l, xₜ, vₜ, grad)

    k = σₒ * kₒ ∘ ScaleTransform(l)
    
    #order 0
        if grad == [0,0]
            return k(xₜ, vₜ)
    
    #order 1
        elseif grad == [1,0]
            return   ForwardDiff.gradient(
                x -> k(x, vₜ)
                , xₜ)
        elseif grad == [0,1]
            return - ForwardDiff.gradient(
                x -> k(x, vₜ)
                , xₜ)
    
    #order 2		
    
        elseif grad == [1,1]
            return - ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , xₜ)
        elseif grad == [2,0] || grad == [0,2]
            return   ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , xₜ)
    
    #order 3		
        elseif grad == [3,0] || grad == [1,2]
            return   ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , x)
            , xₜ)
        elseif grad == [2,1] || grad == [0,3]
            return - ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , x)
            , xₜ)
    
    #order 4
        elseif grad == [4,0] || grad == [2,2] || grad == [0,4]
            return   ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.gradient(
                        x -> k(x, vₜ)
                        , x)
                    , x)
                , x)
            , xₜ)
        elseif grad == [3,1] || grad == [1,3] 
            return - ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.gradient(
                        x -> k(x, vₜ)
                        , x)
                    , x)
                , x)
            , xₜ)
    
        end
    end

end

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


function Posterior(X, xₒ, Target, kₒ, σₒ, l, order)
	dim = size(X,1)
	Kₘₘ⁻¹ = Marginal(X, kₒ, σₒ, l)
	Kₙₘ = Coveriant(X, xₒ, kₒ, σₒ, l, order)
	Meanₚ = Kₙₘ * Kₘₘ⁻¹ * Target	
	return Meanₚ
end
