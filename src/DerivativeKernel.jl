function kernel(kₒ, σₒ , l, xₜ, vₜ, grad)

    k = σₒ * kₒ ∘ ScaleTransform(l)
    d = size(xₜ, 1)
        
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
            kk = - ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , xₜ)
            return reshape(kk, (d, d))
    
        elseif grad == [2,0] || grad == [0,2]
            kk = ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , xₜ)
            return reshape(kk, (d, d))
    
    #order 3		
        elseif grad == [3,0] || grad == [1,2]
            kk =  ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d))
    
        elseif grad == [2,1] || grad == [0,3]
            kk = - ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.gradient(
                    x -> k(x, vₜ)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d))
    
            
    #order 4
        elseif grad == [4,0] || grad == [2,2] || grad == [0,4]
            kk =  ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.gradient(
                        x -> k(x, vₜ)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d))
    
        elseif grad == [3,1] || grad == [1,3] 
            kk =  - ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.gradient(
                        x -> k(x, vₜ)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d))
    
    #order 5
        elseif grad == [5,0] || grad == [3,2] || grad == [1,4]
            kk =   ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x ->ForwardDiff.jacobian(
                    x -> ForwardDiff.jacobian(
                        x -> ForwardDiff.gradient(
                            x -> k(x, vₜ)
                            , x)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d, d))
    
        elseif grad == [0,5] || grad == [2,3] || grad == [4,1] 
            kk = - ForwardDiff.jacobian(
            x ->ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.jacobian(
                        x -> ForwardDiff.gradient(
                            x -> k(x, vₜ)
                            , x)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d, d))
            
    #order 6
        elseif grad == [6,0] || grad == [4,2] || grad == [2,4] || grad == [0,6]
            kk =   ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.jacobian(
                        x -> ForwardDiff.jacobian(
                            x -> ForwardDiff.gradient(
                                x -> k(x, vₜ)
                                , x)
                            , x)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d, d, d))
    
        elseif grad == [5,1] || grad == [1,5] || grad == [3,3] 
            kk = - ForwardDiff.jacobian(
            x -> ForwardDiff.jacobian(
                x -> ForwardDiff.jacobian(
                    x -> ForwardDiff.jacobian(
                        x -> ForwardDiff.jacobian(
                            x -> ForwardDiff.gradient(
                                x -> k(x, vₜ)
                                , x)
                            , x)
                        , x)
                    , x)
                , x)
            , xₜ)
            return reshape(kk, (d, d, d, d, d, d))
        end
    end