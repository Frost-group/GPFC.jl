using Pkg
Pkg.activate(".")
# uncomment the next line if the General registry is not yet installed,
# e.g. if using Julia for the first time.
# Pkg.Registry.add("General")  
Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add("ACEpotentials")

using ACEpotentials

data_file = "notebooks/Si_dia/vasp/Data/d_Si.extxyz"
data = read_extxyz(data_file)

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")

data
train_data = data[1:5:end];
test_data = data[2:10:end];

hyperparams = (elements = [:Si,],
					order = 3,
					totaldegree = 6,
					rcut = 5.5)

hyperparams = (elements = [:Ti, :Al],
					order = 3,
					totaldegree = 6,
					rcut = 5.5,
					Eref = [:Ti => -1586.0195, :Al => -105.5954])
model = ace1_model(; hyperparams...)
@show length_basis(model);


weights = Dict(
        "" => Dict("E" => 1.0, "F" => 1.0 , "V" => 1.0 ));
data_keys = (energy_key = "energy", force_key = "forces", virial_key = "")

solver = ACEfit.BLR()

P = algebraic_smoothness_prior(model; p = 4)    #  (p = 4 is in fact the default)

result = acefit!(train_data, model; solver=solver, prior = P, data_keys...);

show(err)