# RBF Kernel
 function rbf_kernel(x::Vector{Float64}, x_star::Vector{Float64}, σ::Float64, l::Float64)
    r = x .- x_star
    d2 = sum(r .^ 2)
    return σ^2 * exp(-0.5 * d2 / l^2)
end

# 1st-Order Derivative (Gradient)
function kernel_1st_derivative(x, x_star, σ, l)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    return -k_val * r / l^2
end

# 2nd-Order Derivative (Hessian)
function kernel_2nd_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    H = zeros(n, n)
    for i in 1:n
        for j in 1:n
            delta = i == j ? 1.0 : 0.0
            H[i, j] = k_val * (r[i] * r[j] / l^4 - delta / l^2)
        end
    end
    return H
end

# 3rd-Order Derivative (Rank-3 Tensor)
function kernel_3rd_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    T = zeros(n, n, n)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                T[i, j, k] = k_val * (
                    -r[i] * r[j] * r[k] / l^6 +
                    ((i == j ? r[k] : 0.0) +
                     (i == k ? r[j] : 0.0) +
                     (j == k ? r[i] : 0.0)) / l^4
                )
            end
        end
    end
    return T
end

# 4th-Order Derivative (Rank-4 Tensor)
function kernel_4th_derivative(x, x_star, σ, l)
    n = length(x)
    r = x .- x_star
    k_val = rbf_kernel(x, x_star, σ, l)
    T4 = zeros(n, n, n, n)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                for m in 1:n
                    term1 = r[i] * r[j] * r[k] * r[m] / l^8
                    term2 = (
                        (i == j ? r[k]*r[m] : 0.0) +
                        (i == k ? r[j]*r[m] : 0.0) +
                        (i == m ? r[j]*r[k] : 0.0) +
                        (j == k ? r[i]*r[m] : 0.0) +
                        (j == m ? r[i]*r[k] : 0.0) +
                        (k == m ? r[i]*r[j] : 0.0)
                    ) / l^6
                    term3 = (
                        (i == j && k == m ? 1.0 : 0.0) +
                        (i == k && j == m ? 1.0 : 0.0) +
                        (i == m && j == k ? 1.0 : 0.0)
                    ) / l^4
                    T4[i, j, k, m] = k_val * (term1 - term2 + term3)
                end
            end
        end
    end
    return T4
end
