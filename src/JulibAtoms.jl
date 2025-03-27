function Read_JuLIP_Atoms(extxyz_filename::String, num_structure)
    # Read data from the specified extended XYZ file.
    data = read_extxyz(extxyz_filename)

    # Number of structures to process.
    n = num_structure

    # Extract atomic positions for each structure and flatten them.
    X = [reduce(vcat, data[ii].X) for ii in 1:n]

    # Store the equilibrium structure (positions of the first structure).
    equi = X[1]

    # Horizontally Concatenate all atomic positions into a matrix.
    x = hcat(X...)

    # Extract energy data for each structure.
    E = [get_data(data[ii], "energy") for ii in 1:n]

    # Combine all energies into a single vector.
    e = vcat(E...)

    # Extract and flatten forces for each structure (negative gradient of energy).
    ∇E = [-reduce(vcat, get_data(data[ii], "forces")) for ii in 1:n]

    # Vertically concatenate all forces into a single vector.
    ∇e = vcat(∇E...)

    # Combine energies and forces into the target output.
    Target = vcat(e, ∇e)

    # Return the equilibrium structure, positions, target values, and structure count.
    return equi, x, Target, n
end

function quaternion_to_rotation_matrix(q::Quaternion)
    # Make sure q is normalized (if not already). 
    # If `q` is already a unit quaternion, you can skip this step or check norm ~ 1.0.

    # Extract components
    w = real(q)                 # scalar part
    (x, y, z) = imag_part(qr)   # i, j ,k component

    # For a unit quaternion q = w + xi + yj + zk
    # the rotation matrix R is given by (in row-major form):
    #
    # R =  [ w^2 + x^2 - y^2 - z^2    2xy - 2wz           2xz + 2wy
    #        2xy + 2wz              w^2 - x^2 + y^2 - z^2 2yz - 2wx
    #        2xz - 2wy              2yz + 2wx            w^2 - x^2 - y^2 + z^2 ]
    #
    # This can be derived from q * v * q⁻¹ for a vector v.

    return @SMatrix [
        w^2 + x^2 - y^2 - z^2  2x*y - 2w*z            2x*z + 2w*y
        2x*y + 2w*z            w^2 - x^2 + y^2 - z^2  2y*z - 2w*x
        2x*z - 2w*y            2y*z + 2w*x            w^2 - x^2 - y^2 + z^2
    ]
end

function rotate_3n_points(R::AbstractMatrix{<:Real}, coords::AbstractVector{<:Real})
    # Ensure coords has length multiple of 3
    @assert length(coords) % 3 == 0 "coords must have length multiple of 3"
    N = length(coords) ÷ 3
    
    # Reshape coords to a 3 x N matrix 
    #   M[:, i] = ( x_i, y_i, z_i )
    M = reshape(coords, 3, N)
    
    # Apply the rotation matrix: R * M  =>  3 x N
    M_rot = R * M
    
    # Flatten back into 3N-vector
    return vec(M_rot)
end

function Read_JuLIP_Atoms_rotation(extxyz_filename::String, num_structure, num_rotation)
    data = read_extxyz(extxyz_filename)

    n = num_structure

    # Extract atomic positions for each structure and flatten them
    X11 = [reduce(vcat, data[ii].X) for ii in 1:n]
    X0 = [reduce(vcat, data[1].X) for _ in 1:n]
    X1 = X11 .- X0  # Subtract reference positions
    equi = X1[1]
    # Extract energies and forces for each structure
    E1 = [get_data(data[ii], "energy") for ii in 1:n]
    ∇E1 = [-reduce(vcat, get_data(data[ii], "forces")) for ii in 1:n]

    # Initialize storage vectors
    X = [X1[1]]  # Include the first structure
    E = [E1[1]]
    ∇E = [∇E1[1]]

# Loop over structures and rotations
    for ii in 2:n
        # Add the original structure data
        push!(X, X1[ii])
        push!(E, E1[ii])
        push!(∇E, ∇E1[ii])

        # Generate rotations
        for jj in 1:num_rotation
            #Random.seed!(jj)  # Seed random number generator

            # Generate a random quaternion and normalize it
            qr = normalize(Quaternion(randn(), randn(), randn(), randn()))

            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qr)

            # Rotate positions and forces, append results
            push!(X, rotate_3n_points(R, X1[ii]))
            push!(E, E1[ii])  # Energy remains unchanged for rotations
            push!(∇E, rotate_3n_points(R, ∇E1[ii]))
        end
    end

    # Concatenate final results
    X_final = hcat(X...)  # Combine all vectors in X into a matrix
    E_final = E  # Energies are stored as a vector
    ∇E_final = vcat(∇E...)  # Combine all force vectors row-wise
    Target_final = vcat(E_final, ∇E_final)  # Combine energy and force data


    # Return the equilibrium structure, positions, target values, and structure count.
    return equi, X_final, Target_final, n, num_rotation
end
