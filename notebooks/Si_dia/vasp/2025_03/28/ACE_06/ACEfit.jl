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
    data= read_extxyz("d_Si.extxyz")
end

train_data = data[1:10:end]
test_data = data[2:2:end]
train_data = data[1:10]
test_data = data[11:509]

function get_Pr(p, q, r0, rcut)
    basis = ACE1x.ace_basis(; elements = [:Si,],
	order = 2,
	totaldegree = 10,
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


begin
	r_cut = 20.0
	rdf_tiny = ACEpotentials.get_rdf(train_data, r_cut; rescale = true)
	plt_rdf_1 = stephist(rdf_tiny[(:Si,:Si)], 
		bins=150, label = "rdf",
		title="Bead_tiny_dataset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+3, 2e+3, 3e+3, 4e+3, 5e+3], xlims=(0,20.5), ylims=(0.0, 5e+3),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	#vline!(rnn(:Si)*[1.0, 4.65, 5.55], label = "r1, r2, ...", lw=2, color = "black")
	
	rdf = ACEpotentials.get_rdf(test_data, r_cut; rescale = true);
	plt_rdf_2 = stephist(rdf[(:Si,:Si)],
		bins=150, label = "rdf",
		title="Bead_dataset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1.5e+3, 3.0e+3, 4.5e+3, 6.0e+3, 7.5e+3], xlims=(0,20.5), ylims=(0.0, 7.5e+3),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	#vline!(rnn(:Si)*[1.0, 4.65, 5.55], label = "r1, r2, ...", lw=2, color = "black")
	
	r0 = rnn(:Si); rcut = 20.0
	rr, Pr = get_Pr(1, 4, r0, rcut)
	plt = plot()
	for n = 1:length(Pr) 
	    plot!(plt, rr, Pr[n], lw=2, label = "P_$n", 
			xlabel = L"r[\AA]", ylabel = "RBF", legend=false, xlims=(0,20.5))
	end
	vline!([r0,], lw=2, ls = :dash, c= :black, label = "r0")
    plot!(plt_rdf_1, plt_rdf_2, plt, layout=(3,1), size=(900,900))
	savefig("_RDF_RBF_Si.png") 
	#savefig("RDF_bead.png") 
end

begin
	p = 1
	q = 4
	model1 = acemodel(elements = [:Si,],
	        order = 2,
	        totaldegree = 10, 
			r0 = rnn(:Si),
	        rcut = 6.0,
			#transform = (:agnesi, p, q),
			pair_transform = (:agnesi, p, q),
			pair_envelope = (:x, 2, 2),
			#envelope = (:x, 2, 2),
	        #Eref = [:Si => 0.0, :O => 0.0]
	)
	@show length(model1.basis);
end

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
             title = @sprintf("Structures Used In Training:  %i out of %i\n", length(train_dataset), length(test_data)) *
                     @sprintf("RMSE (MAE) For Entire Dataset:  %.0f (%.0f) meV/atom", 1000*rmse, 1000*mae),
             titlefontsize = 8,
             yerror = model_std,
             xlabel="Energy [eV/atom]", xlims=(-5.424,-5.42),
             ylabel="Model Energy [eV/atom]", ylims=(-5.424,-5.42),
             aspect_ratio = :equal, color="#F0BB62")

    model_energies = [energy(model.potential,atoms)/length(atoms) for atoms in train_dataset]
    scatter!(extract_energies(train_dataset), model_energies;
             label="training set", color="#064635")
savefig("fitting_Si.png") 
end;

assess_model(model1, train_data)