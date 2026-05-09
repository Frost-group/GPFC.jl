using Pkg
#cd("/Users/kk419/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2025_03/28/ACE_06")
cd("C:/Users/Keerati/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2025_03/28/ACE_06/")
Pkg.activate(".")
Base.active_project()
Pkg.instantiate()
begin# add and load general packages used in this notebook.
	Pkg.add(["LaTeXStrings", "MultivariateStats", "Plots", "Suppressor"
         ])
	using LaTeXStrings, MultivariateStats, Plots, Printf, Statistics, Suppressor
end;
begin
	#Pkg.Registry.add("General")  # only needed when installing Julia for the first time
	Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
	Pkg.add(PackageSpec(name="ACEpotentials", version="0.6.7"))
end

using LaTeXStrings, MultivariateStats, Plots, Printf,
	Statistics, Suppressor, ACEpotentials

begin	
	cd("C:/Users/Keerati/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2025_03/28/ACE_06/")
    data= read_extxyz("d_Si.extxyz")
end



if !isfile("Si_dataset.xyz")
    download("https://www.dropbox.com/scl/fi/z6lvcpx3djp775zenz032/Si-PRX-2018.xyz?rlkey=ja5e9z99c3ta1ugra5ayq5lcv&st=cs6g7vbu&dl=1",
         "Si_dataset.xyz");
end

Si_dataset = read_extxyz("Si_dataset.xyz");


train_data = data[1:10:end]
test_data = data[2:2:end]
train_data = data[1:50]
test_data = data[11:509]


Si_scale = [1.0, 1.633, 1.915, 2.31, 2.52]
begin
	p = 1
	q = 4
	model1 = acemodel(elements = [:Si,],
	        order = 2,
	        totaldegree = 5,
			r0 = rnn(:Si),
	        rcut = 6.0,
			#transform = (:agnesi, p, q),
			pair_transform = (:agnesi, p, q),
			pair_envelope = (:r, 2, 2),
			#envelope = (:x, 2, 2),
	        #Eref = [:Si => 0.0, :O => 0.0]
	)
	@show length(model1.basis);
end
n = 4
train_data = data[2:n]
test_data = data[n+1:500]

data_keys1 = (energy_key = "energy", force_key = "forces", virial_key = "")
solver = ACEfit.BLR(committee_size=500, factorization=:svd)
acefit!(model1, train_data; solver=solver, data_keys1...);


begin
	@info("Training Errors")
	ACEpotentials.linear_errors(train_data, model1; data_keys1...);
	
	@info("Test Error")
	ACEpotentials.linear_errors(test_data, model1; data_keys1...);
end


begin
	function extract_energies(dataset)
	    energies = []
	    for atoms in dataset
	        for key in keys(atoms.data)
	            if lowercase(key) == "energy"
	                push!(energies, atoms.data[key].data/length(atoms))
	            end
	        end
	    end
	    return energies
	end;

	bead_tiny_energies = extract_energies(train_data)
	bead_energies = extract_energies(test_data)

	
	GC.gc()
end

begin
	function extract_forces(dataset)
	    forces = []
	    for atoms in dataset
	        for key in keys(atoms.data)
	            if lowercase(key) == "forces"
	                push!(forces, atoms.data[key])
	            end
	        end
	    end
	    return forces
	end;

	function extract_forces2(forces)
	    forces2 = []
	    for j in forces
	    	push!(forces2, j.data)
	    end
	    return forces2
	end;

	function extract_energies(dataset)
	    energies = []
	    for atoms in dataset
	        for key in keys(atoms.data)
	            if lowercase(key) == "energy"
	                push!(energies, atoms.data[key].data/length(atoms))
	            end
	        end
	    end
	    return energies
	end;

	bead_tiny_energies = extract_energies(train_data)
	bead_energies = extract_energies(test_data)

	bead_tiny_forces = extract_forces(train_data)
	bead_forces = extract_forces(test_data)

	bead_tiny_forces2 = extract_forces2(bead_tiny_forces)
	bead_forces2 = extract_forces2(bead_forces)
	GC.gc()
end

model_forces = []
model_f_std = []
for atoms in test_data
        forc, co_forc = ACE1.co_forces(model1.potential, atoms)
        push!(model_forces, forc)
        push!(model_f_std, std.(co_forc))
end

model_energies = []
model_e_std = []
    for atoms in test_data
        ene, co_ene = ACE1.co_energy(model1.potential, atoms)
        push!(model_energies, ene/length(atoms))
        push!(model_e_std, std(co_ene/length(atoms)))
    end

model_forces

bead_forces2 
#fieldnames(typeof(j))


function rmse_mae_all_scalars(a, b)
    @assert length(a) == length(b) "Outer vectors must have same length"

    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n = 0

    for (va, vb) in zip(a, b)
        @assert length(va) == length(vb) "Inner vectors must have same length"
        for (sa, sb) in zip(va, vb)
            @assert length(sa) == length(sb) == 3
            @inbounds for i in 1:3
                diff = sa[i] - sb[i]
                sq_err_sum += diff^2
                abs_err_sum += abs(diff)
                n += 1
            end
        end
    end

    rmse = sqrt(sq_err_sum / n)
    mae  = abs_err_sum / n
    return rmse, mae
end

rmse, mae = rmse_mae_all_scalars(model_forces, bead_forces2 )
println("RMSE = $rmse")
println("MAE  = $mae")



function assess_model_forces(model, train_dataset)

    plot([-6,-4], [-6,-4]; lc=:black, label="")

    model_forces = []
    model_std = []
    for atoms in test_data
        forc, co_forc = ACE1.co_forces(model.potential, atoms)
        push!(model_forces, forc)
        push!(model_std, std.(co_forc))
    end
	
    rmse, mae = rmse_mae_all_scalars(model_forces, bead_forces2 )

#println(bead_energies)
println("_____________________________")
println(model_forces)
    scatter!(bead_forces2, model_forces;
             label="full dataset",
             title = @sprintf("d-Si Structures Used In Training:  %i out of %i\n", length(train_data), length(test_data)) *
                     @sprintf("RMSE (MAE) For Entire Dataset:  %.0f (%.0f) meV/Å", rmse, mae),
             titlefontsize = 8,
             yerror = model_std,
             xlabel="Force [eV/"*L"\text{\AA}"*"]", xlims=(-5.424,-5.42),
             ylabel="Model Force [eV/"*L"\text{\AA}"*"]", ylims=(-5.424,-5.42),
             aspect_ratio = :equal, color="#F0BB62")

    model_forces = [forces(model.potential,atoms)/length(atoms) for atoms in train_dataset]
    scatter!(extract_forces2(train_dataset), model_forces;
             label="training set", color="#064635")
#savefig("fitting_Si_.png") 

end;

model_energies = []
model_std = []
for atoms in test_data
    ene, co_ene = ACE1.co_energy(model1.potential, atoms)
    push!(model_energies, ene/length(atoms))
    push!(model_std, std(co_ene/length(atoms)))
end
	
rmse = sqrt(sum((model_energies - bead_energies).^2)/length(test_data))
mae = sum(abs.(model_energies - bead_energies))/length(test_data)


function assess_model(model, train_dataset)
	gr(dpi=1200)
    plot([-0.5,0.5], [-0.5,0.5]; lc=:black, label="")

    model_energies = []
    model_std = []
    for atoms in bead_dataset
        ene, co_ene = ACE1.co_energy(model.potential, atoms)
        push!(model_energies, ene/length(atoms))
        push!(model_std, std(co_ene/length(atoms)))
    end
	
    rmse = sqrt(sum((model_energies - bead_energies).^2)/length(bead_dataset))
    mae = sum(abs.(model_energies - bead_energies))/length(bead_dataset)
	
#println(bead_energies)
println("_____________________________")	
println(model_energies)
    scatter!(bead_energies*4, model_energies*4;
             label="full dataset",
             title = @sprintf("Structures Used In Training:  %i out of %i\n", length(train_data), length(test_data)+length(train_data)) *
                     @sprintf("RMSE (MAE) For Entire Dataset:  %.0f (%.0f) μeV/molecule", 1e+6*rmse*4, 1e+6*mae*4),
             titlefontsize = 8,
             yerror = model_std,
           	 xlabel="Energy [μeV/molecule]", xlims=(-7e-5, 11e-5),
             ylabel="Model Energy [μeV/molecule]", ylims=(-7e-5, 11e-5),
             aspect_ratio = :equal, color="#F0BB62")

    model_energies = [energy(model.potential,atoms)/length(atoms) for atoms in train_data]
    scatter!(extract_energies(train_data)*4, model_energies*4;
            label="training set", color="#064635",
			size=(500,500), fontsize=12,
			xticks = ([-6e-5, -4e-5, -2e-5, 0, 2e-5, 4e-5, 6e-5, 8e-5, 10e-5],
            		["-60", "-40","-20", "0", "20", "40", "60", "80" , "100"]),
			yticks = ([-6e-5, -4e-5, -2e-5, 0, 2e-5, 4e-5, 6e-5, 8e-5, 10e-5],
            		["-60", "-40","-20", "0", "20", "40", "60", "80", "100"]),
			)
#savefig("fitting_NH_CH_od$od"*"_td$td"*"_p$p"*"_q$q"*"_rcut75_N3000_2.png") 
#savefig("fitting_NH_CSi_od$od"*"_td$td"*"_p$p"*"_q$q"*"_rcut$rcut"*"_N$n_train"*"_rand_mol.png") 
	println(1e+6*rmse*4)
	println(1e+6*mae*4)
end;

assess_model(model1, bead_tiny_dataset)
