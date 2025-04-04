using Pkg
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
train_data = data[1:10]
test_data = data[11:509]

function get_Pr(p, q, r0, rcut, td)
    basis = ACE1x.ace_basis(; elements = [:Si,],
	order = 2,
	totaldegree = td,
	r0 = r0,
	rcut = rcut, 
    pair_transform = (:agnesi, p, q),
    pair_envelope = (:r, 2, 2)
	)
    Pr = basis.BB[1].J[1,1]
    env = Pr.envelope
    rr = range(0.01, rcut, length=400)
    vals = ACE1.evaluate.(Ref(Pr), rr)
    vals2 = [ [vals[j][i] / ACE1.evaluate(Pr.envelope, rr[j]) for j = 1:length(vals) ] 
              for i = 1:length(vals[1]) ]
    return rr, vals2 
end


Si_scale = [1.0, 1.633, 1.915, 2.31, 2.52]
begin
	r_cut = 6.0; a = 2
	rdf_tiny = ACEpotentials.get_rdf(train_data, r_cut; rescale = true)
	plt_rdf_1 = stephist(rdf_tiny[(:Si,:Si)], 
		bins=150, label = "rdf",
		title="d-Si_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0,6.5), ylims=(0.0, 5e+2/a),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Si)*Si_scale, label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	b = 0.02
	rdf = ACEpotentials.get_rdf(test_data, r_cut; rescale = true);
	plt_rdf_2 = stephist(rdf[(:Si,:Si)],
		bins=150, label = "rdf",
		title="d-Si_testset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/b, 2e+2/b, 3e+2/b, 4e+2/b, 5e+2/b], xlims=(0,6.5), ylims=(0.0, 5e+2/b),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Si)*Si_scale, label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	td = 5 
	r0 = rnn(:Si); rcut = 6.0; s = 1
	rr, Pr = get_Pr(1, 4, r0*Si_scale[s], rcut, td)

	r0 = rnn(:Si); rcut = 5.0; s1 = mean(Si_scale[1:3])
	rr1, Pr1 = get_Pr(1, 4, r0*s1, rcut, td)

	r0 = rnn(:Si); rcut = 3.0; s2 = mean(Si_scale[1:5])
	rr2, Pr2 = get_Pr(1, 4, r0*s2, rcut, td)
	plt = plot()
	for n = 1:length(Pr)
	    plot!(plt, rr2, Pr2[n], lw=2, label = "P_$n", 
			xlabel = L"r[\AA]", ylabel = "RBF", legend=false, xlims=(0,6.5),
			color= "#C64756")
		plot!(plt, rr1, Pr1[n], lw=2, label = "P_$n", 
			xlabel = L"r[\AA]", ylabel = "RBF", legend=false, xlims=(0,6.5),
			color= "#6E9A50")
		plot!(plt, rr, Pr[n], lw=2, label = "P_$n", 
			xlabel = L"r[\AA]", ylabel = "RBF", legend=false, xlims=(0,6.5),
			color= "#F0BB62")
	end
	vline!([r0*s2,], lw=2, ls = :dash, c= "#F0BB62", label = "r0")
	vline!([r0*s1,], lw=2, ls = :dash, c= "#6E9A50", label = "r0")
	vline!([r0*s,], lw=2, ls = :dash, c= "#C64756", label = "r0")
    plot!(plt_rdf_1, plt_rdf_2, plt, layout=(3,1), size=(900,600))
	#savefig("_RDF_RBF_Si.png") 
	#savefig("RDF_bead.png") 
end


Si_scale = [1.0, 1.633, 1.915, 2.31, 2.52]
begin
	p = 1
	q = 4
	model1 = acemodel(elements = [:Si,],
	        order = 2,
	        totaldegree = 5,
			r0 = rnn(:Si)* mean(Si_scale[1:5]),
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


function assess_model(model, train_dataset)

    plot([-6,-4], [-6,-4]; lc=:black, label="")

    model_energies = []
    model_std = []
    for atoms in test_data
        ene, co_ene = ACE1.co_energy(model.potential, atoms)
        push!(model_energies, ene/length(atoms))
        push!(model_std, std(co_ene/length(atoms)))
    end
	
    rmse = sqrt(sum((model_energies - bead_energies).^2)/length(test_data))
    mae = sum(abs.(model_energies - bead_energies))/length(test_data)

#println(bead_energies)
println("_____________________________")	
println(model_energies)
    scatter!(bead_energies, model_energies;
             label="full dataset",
             title = @sprintf("d-Si Structures Used In Training:  %i out of %i\n", length(train_dataset), length(test_data)) *
                     @sprintf("RMSE (MAE) For Entire Dataset:  %.0f (%.0f) meV/atom", 1000*rmse, 1000*mae),
             titlefontsize = 8,
             yerror = model_std,
             xlabel="Energy [eV/atom]", xlims=(-5.424,-5.42),
             ylabel="Model Energy [eV/atom]", ylims=(-5.424,-5.42),
             aspect_ratio = :equal, color="#F0BB62")

    model_energies = [energy(model.potential,atoms)/length(atoms) for atoms in train_dataset]
    scatter!(extract_energies(train_dataset), model_energies;
             label="training set", color="#064635")
#savefig("fitting_Si_.png") 
end;

assess_model(model1, train_data)
  