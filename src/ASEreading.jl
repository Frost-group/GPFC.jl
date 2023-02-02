function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, num)
	feature = (
		CSV.File(
			FileFeature
		)|> Tables.matrix
	)[
		begin:3:end
		,2:num+1
	]
	equi = feature[:,1]
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (
		CSV.File(
			FileEnergy
		)|> Tables.matrix
	)[
		begin:num
		,2
	]

	force = -reshape(
	(
		CSV.File(
			FileForce
		)|> Tables.matrix
	)[
		begin:3:end
		,2:num+1
	]
	, (dim*num,1)
	)

	Target = vcat(
		energy
		, reshape(
			force
			, (dim*num,1)
		)
	)

	return equi, feature, energy, force, Target
end