function kernel(k, xₜ, vₜ, grad)

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