function ASEFeatureTarget(FileFeature, FileEnergy, FileForce, numt::Int64, dimA::Int64)
	a  = 4 - dimA
	feature = (
		CSV.File(
			FileFeature
		)|> Tables.matrix
	)[
		begin:a:end
		,2:numt+1
	]
	equi = feature[:,1]
	dim = size(feature,1)
	num = size(feature,2)
	
	energy = (
		CSV.File(
			FileEnergy
		)|> Tables.matrix
	)[
		begin:numt
		,2
	]

	force = -reshape(
	(
		CSV.File(
			FileForce
		)|> Tables.matrix
	)[
		begin:a:end
		,2:numt+1
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