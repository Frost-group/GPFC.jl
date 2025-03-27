using Pkg
Pkg.activate(".")
# uncomment the next line if the General registry is not yet installed,
# e.g. if using Julia for the first time.
# Pkg.Registry.add("General")  
Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add("ACEpotentials")

# uncomment the next line if the General registry is not yet installed,
# e.g. if using Julia for the first time.
# Pkg.Registry.add("General")
Pkg.add(["LaTeXStrings", "MultivariateStats", "Plots", "PrettyTables", 
         "Suppressor", "ExtXYZ", "Unitful", "Distributed", "AtomsCalculators", 
         ])

using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf, 
		 Statistics, Suppressor, ExtXYZ, Unitful
		 
using ACEpotentials
using ACE1

Pkg.status()

data_file = "notebooks/Si_dia/vasp/Data/d_Si.extxyz"
data = ExtXYZ.load(data_file)

train_data = data[1:10:end]
test_data = data[2:2:end]



hyperparams = (elements = [:Si,],
					order = 3,
					totaldegree = 6,
					rcut = 5.5)


model = ace1_model(; hyperparams...)
@show length_basis(model);


weights = Dict(
        "" => Dict("E" => 1.0, "F" => 1.0 , "V" => 1.0 ));
data_keys = (energy_key = "energy", force_key = "forces", virial_key = "")

solver = ACEfit.BLR()

P = algebraic_smoothness_prior(model; p = 4)    #  (p = 4 is in fact the default)

result = acefit!(train_data, model; 
				solver = ACEfit.BLR(committee_size = 30, factorization = :svd),
				energy_key = "energy", force_key = "forces", verbose = false);
