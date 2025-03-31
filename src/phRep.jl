"""
    phonon_Γ(natom::Int64, feature, force, Target)

Calculate the derivative of kernel function 'k'.
A standard (or compositing) kernel function requires to be defined by using KernelFunctions.jl.
This will turn the defined kernel function and two different atomistic
representation vectors (features) to the derivative of the kernel of those two vectors where the
derivative order can be specified by variable grad. 
"""
function phonon_Γ(natom::Int64, feature, force, Target)
	dim = 3 * natom
	SupercellSize = Int(size(feature, 1)/dim)
	ndata = size(feature, 2)

	feature_ph = zeros((dim, ndata))
	for jj in 1:ndata
		for kk in 1:SupercellSize
			if kk == 1
				feature_ph[1:dim, jj] = feature[1:dim, jj]
			else
				feature_ph[1:dim, jj] = feature_ph[1:dim, jj] + feature[dim*(kk-1)+1:dim*kk, jj]
			end 
		end
	end	
	equi_ph = feature_ph[1:dim,1]

	force_r = reshape(force, (dim*SupercellSize, ndata))
	force_ph = zeros((dim, ndata))
	for jj in 1:ndata
		for kk in 1:SupercellSize
			if kk == 1
				force_ph[1:dim,jj] = force_r[1:dim,jj]
			else
				force_ph[1:dim,jj] =force_ph[1:dim,jj] + force_r[dim*(kk-1)+1:dim*kk,jj]
			end 
		end
	end
	Target_ph = zeros(((1+dim)*ndata))
	Target_ph[1:ndata] = Target[1:ndata]
	Target_ph[1+ndata:(1+dim)*ndata] = reshape(force_ph,(dim*ndata,1))
	return equi_ph, feature_ph, Target_ph, force_ph 
end

function phonon_dft(x::Vector{Float64}, q::Vector{Float64}, R::Array{Float64}, masses::Vector{Float64})
    numAtom = Int(size(masses, 1)/3)
    N3 = size(R, 2)
    d = zeros(ComplexF64, 3* numAtom)
    q_atom = repeat(q,  numAtom)
    x2d = reshape(x, 3*numAtom, N3)

    for l in 1:N3
        R_lB = R[:, l]       # equilibrium position of atom B in cell l
        R_0A = R[:, 1]       # reference position (atom A in cell 0)

        phase = exp(-im * dot(q_atom, R_lB - R_0A))
        #d .+= sqrt.(masses) .* x2d[:,l] * phase
		d .+= x2d[:,l] * phase
    end

    d_realimag = vcat(real(d), imag(d))  # shape (2, 3m)
    return d_realimag
end