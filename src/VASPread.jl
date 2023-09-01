function VASP_input(vasph5)
	path = vasph5  # string, Name of vasp out put vas output files (HDF5)
	file = h5open(path, "r")

	# Ionic info
	input_position_ions = read(file["input/poscar/position_ions"])'
	type_ions = read(file["input/poscar/ion_types"]) # string, name of elements.
	#atomicnumber_ions = [82 , 52]
	atomicnumber_ions = [14]
	
	number_iontypes = read(file["input/poscar/number_ion_types"])
	number_ions = sum(number_iontypes)

	energy = read(file["intermediate/ion_dynamics/energies"])
	TOTEN = energy[1,:] #ion-electron
	EKIN = energy[2,:] #kinetic energy
	EKIN_LAT = energy[3,:] #kinetic of lattice
	TEIN = energy[4,:] #temperature
	ES = energy[5,:] #nose potential
	EPS = energy[6,:] #nose kinetic
	ETOTAL = energy[7,:] #total energy
	
	datasize = size(energy,2) #iteration
	forces = read(file["intermediate/ion_dynamics/forces"])
	intermediate_position_ions = read(file["intermediate/ion_dynamics/position_ions"])
	lattice_vectors = read(file["intermediate/ion_dynamics/lattice_vectors"])#

	feature = reshape(intermediate_position_ions, (3 * number_ions, datasize))
	forceset = reshape(forces, (3 * number_ions, datasize))

	rev_feature = zeros(size(feature))
	rev_forces = zeros(size(forceset))
	rev_energy = zeros(datasize)

	for ii in 1:datasize
		rev_feature[:, ii] = feature[:, (datasize+1) - ii]
		rev_forces[:, ii] = forceset[:, (datasize+1) - ii]
		rev_energy[ii] = ETOTAL[(datasize+1) - ii]
	end
	
	Target = zeros((1+3*number_ions, datasize))
	Target[1,:] = rev_energy
	Target[2:1+3*number_ions,:] = rev_forces

	return rev_feature, rev_energy, rev_forces, Target 
end;