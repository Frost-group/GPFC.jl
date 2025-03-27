# Define the kernel function
function kernel(x, x_star, sigma, l)
    # Compute the Euclidean distance squared
    d_squared = sum((x .- x_star).^2)
    
    # Compute the kernel function
    k = sigma * exp(-0.5 * d_squared / l^2)
    
    return k
end

# Compute the gradient (first derivative) of the kernel function
function kernel_derivative(x, x_star, sigma, l)
    # Compute the Euclidean distance squared
    d_squared = sum((x .- x_star).^2)
    
    # Compute the kernel function
    k = sigma * exp(-0.5 * d_squared / l^2)
    
    # Compute the gradient with respect to x
    grad_k = - (sigma / l^2) * (x .- x_star) * exp(-0.5 * d_squared / l^2)
    
    return grad_k
end

# Compute the Hessian (second derivative) of the kernel function
function kernel_hessian(x, x_star, sigma, l)
    # Compute the Euclidean distance squared
    d_squared = sum((x .- x_star).^2)
    
    # Compute the kernel function
    k = sigma * exp(-0.5 * d_squared / l^2)
    
    # Initialize the Hessian matrix (size: length of x)
    n = length(x)
    H = zeros(n, n)
    
    # Compute the Hessian matrix
    for i in 1:n
        for j in 1:n
            if i == j
                # Diagonal elements
                H[i, j] = - (sigma / l^2) * exp(-0.5 * d_squared / l^2) * (1 - ((x[i] - x_star[i])^2 / l^2))
            else
                # Off-diagonal elements
                H[i, j] = - (sigma / l^2) * exp(-0.5 * d_squared / l^2) * ((x[i] - x_star[i]) * (x[j] - x_star[j]) / l^2)
            end
        end
    end
    
    return H
end