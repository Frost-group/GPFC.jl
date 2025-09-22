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
	cd("C:/Users/Keerati/Documents/GitHub/GPFC.jl/notebooks/PbTe/vasp/2025_04/03/")
    data= read_extxyz("PbTe_filtered.extxyz")
end



if !isfile("Si_dataset.xyz")
    download("https://www.dropbox.com/scl/fi/z6lvcpx3djp775zenz032/Si-PRX-2018.xyz?rlkey=ja5e9z99c3ta1ugra5ayq5lcv&st=cs6g7vbu&dl=1",
         "Si_dataset.xyz");
end

Si_dataset = read_extxyz("Si_dataset.xyz");


train_data = data[1:50]
test_data = data[11:497]

function get_Pr(p, q, r0, rcut, td)
    basis = ACE1x.ace_basis(; elements = [:Pb,:Te],
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

begin
	gr(dpi=1200) 
	r_cut = 6.5; a = .2
	plt = plot()
	rdf_tiny = ACEpotentials.get_rdf(train_data, r_cut; rescale = true)
	plt_PbPb = stephist(rdf_tiny[(:Pb,:Pb)].-1.1, 
		bins=150, label = "rdf",
		title="PbPb_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Pb)*[1.0, 21.633, 21.915, 2.31, 2.52], label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	plt_PbTe = stephist(rdf_tiny[(:Pb,:Te)].+0.43, 
		bins=150, label = "rdf",
		title="PbTe_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a/2, 2e+2/a/2, 3e+2/a/2, 4e+2/a/2, 5e+2/a/2], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a/2),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!((rnn(:Pb)+rnn(:Te))/2*[1.0, 1.633, 1.915, 2.31, 2.52], label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	plt_TeTe = stephist(rdf_tiny[(:Te,:Te)].-0.7, 
		bins=150, label = "rdf",
		title="TeTe_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Te)*[1.0, 21.633, 1.915, 2.31, 2.52], label = "r1, r2, ...", lw=1, color = "black", ls = :dash)

	plot(plt_PbPb, plt_PbTe, plt_TeTe, layout = (3,1), size = (900,600), left_margin = 6Plots.mm)
end
Si_scale = [1.0, 1.633, 1.915, 2.31, 2.52]
begin
	gr(dpi=1200) 
	r_cut = 6.5; a = .2
	rdf_tiny = ACEpotentials.get_rdf(train_data, r_cut; rescale = true)
	plt = plot()
	stephist!(rdf_tiny[(:Pb,:Pb)].-1.1, 
		bins=100, label = "PbPb RDF",
		title="PbPb_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Pb)*[1.0, 21.633, 21.915, 2.31, 2.52], label = "PbPb "*L"r_0", lw=2, color = "#F0BB62", ls = :dash)
	
	stephist!(rdf_tiny[(:Pb,:Te)].+0.43, 
			bins=100, label = "PbTe RDF",
			title="PbTe_trainset", titlefontsize=10,
			xlabel = L"r[\AA]", ylabel = "RDF",
			yticks = [1e+2/a/2, 2e+2/a/2, 3e+2/a/2, 4e+2/a/2, 5e+2/a/2], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a/2),
			size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#6E9A50")
	vline!((rnn(:Pb)+rnn(:Te))/2*[1.0, 1.633, 1.915, 2.31, 2.52], label = "PbTe "*L"r_0", lw=2, color = "#6E9A50", ls = :dash)
		
	stephist!(rdf_tiny[(:Te,:Te)].-0.7, 
			bins=100, label = "TeTe RDF",
			title="Trainset", titlefontsize=10,
			xlabel = L"r[\AA]", ylabel = "RDF",
			yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0.,r_cut+0.5), ylims=(0.0, 5e+2/a),
			size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#C64756")
	vline!(rnn(:Te)*[1.0, 21.633, 1.915, 2.31, 2.52], label = "TeTe "*L"r_0", lw=2, color = "#C64756", ls = :dash)
	
	td = 5 
	r0 = (rnn(:Pb)+rnn(:Te))/2; rcut = 6.5; s = 1
	rr, Pr = get_Pr(1, 4, r0*Si_scale[s], rcut, td)

	plt1 = plot()
	for n = 1:length(Pr)
		plot!(plt1, rr, Pr[n], lw=2, label = "", 
			xlabel = L"r[\AA]", ylabel = "RBF", legend=false, xlims=(0,r_cut+0.5),
			color= "black",)
	end
	vline!([r0*s,], lw=2, ls = :dash, c= "black", label = "PbTe "*L"r_0", legend=true)

	plot!(plt, plt1, layout=(2,1), size=(800,400))
	savefig("_RDF_RBF_PbTe.png") 
end

begin
	r_cut_adf_Pb = 1.25 * rnn(:Pb) #
	r_cut_adf_Te = 1.25 * rnn(:Te) #
	r_cut_adf_PbTe = 1.25 * (rnn(:Te)+rnn(:Pb))/2 #
	
	eq_angle = [45, 60, 90, 120, 135]* pi/180 # radians
	
	adf_tiny_Pb = ACEpotentials.get_adf(test_data, r_cut_adf_Pb);
	plt_adf_1 = stephist(adf_tiny_Pb, bins=50, label = "adf", yticks = [], c = 3,
	                    title = "Pb", titlefontsize = 10,
	                    xlabel = L"\theta", ylabel = "ADF Pb",
	                    xlims = (0, π), size=(400,200), left_margin = 2Plots.mm, 
						fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!([ eq_angle[3],], label = "90˚", lw=1, color = "black")


	adf_tiny_Te = ACEpotentials.get_adf(test_data, r_cut_adf_Te);
	plt_adf_2= stephist(adf_tiny_Te, bins=50, label = "adf", yticks = [], c = 3,
	                    title = "Te", titlefontsize = 10,
	                    xlabel = L"\theta", ylabel = "ADF Te",
	                    xlims = (0, π), size=(400,200), left_margin = 2Plots.mm,
						fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!([ eq_angle,], label = "45˚, 60˚, 90˚, 120˚, 135˚", lw=1, color = "black")
	
	adf_tiny_PbTe = ACEpotentials.get_adf(test_data, r_cut_adf_PbTe);
	plt_adf_3= stephist(adf_tiny_PbTe, bins=50, label = "adf", yticks = [], c = 3,
	                    title = "PbTe", titlefontsize = 10,
	                    xlabel = L"\theta", ylabel = "ADF PbTe",
	                    xlims = (0, π), size=(400,200), left_margin = 2Plots.mm,
						fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!([ eq_angle,], label = "45˚, 60˚, 90˚, 120˚, 135˚", lw=1, color = "black")

	plot(plt_adf_1, plt_adf_2, plt_adf_3, layout=(3,1), size=(600,600))
end

begin
	r_cut = 6.0; a = 2
	rdf_tiny = ACEpotentials.get_rdf(train_data, r_cut; rescale = true)
	plt_rdf_1 = stephist(rdf_tiny[(:Si,:Si)], 
		bins=150, label = "rdf",
		title="d-Si_trainset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/a, 2e+2/a, 3e+2/a, 4e+2/a, 5e+2/a], xlims=(0,6.5), ylims=(0.0, 5e+2/a),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Si)*[1.0, 1.633, 1.915, 2.31, 2.52], label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	b = 0.02
	rdf = ACEpotentials.get_rdf(test_data, r_cut; rescale = true);
	plt_rdf_2 = stephist(rdf[(:Si,:Si)],
		bins=150, label = "rdf",
		title="d-Si_testset", titlefontsize=10,
		xlabel = L"r[\AA]", ylabel = "RDF",
		yticks = [1e+2/b, 2e+2/b, 3e+2/b, 4e+2/b, 5e+2/b], xlims=(0,6.5), ylims=(0.0, 5e+2/b),
		size=(400,200), left_margin = 2Plots.mm, fill=true, fillalpha=0.5, color= "#F0BB62")
	vline!(rnn(:Si)*[1.0, 1.633, 1.915, 2.31, 2.52], label = "r1, r2, ...", lw=1, color = "black", ls = :dash)
	
	td = 6
	r0 = rnn(:Si); rcut = 6.0; s = 4
	rr, Pr = get_Pr(1, 4, r0*Si_scale[s], rcut, td)

	r0 = rnn(:Si); rcut = 5.0; s1 = 2
	rr1, Pr1 = get_Pr(1, 4, r0*Si_scale[s1], rcut, td)

	r0 = rnn(:Si); rcut = 3.0; s2 = 1
	rr2, Pr2 = get_Pr(1, 4, r0*Si_scale[s2], rcut, td)
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
	vline!([r0*Si_scale[s],], lw=2, ls = :dash, c= "#F0BB62", label = "r0")
	vline!([r0*Si_scale[s1],], lw=2, ls = :dash, c= "#6E9A50", label = "r0")
	vline!([r0*Si_scale[s2],], lw=2, ls = :dash, c= "#C64756", label = "r0")
    plot!(plt_rdf_1, plt_rdf_2, plt, layout=(3,1), size=(900,600))
	#savefig("_RDF_RBF_Si.png") 
	#savefig("RDF_bead.png") 
end

Si_scale = [1.0, 1.633, 1.915, 2.31, 2.52]
begin
	p = 1
	q = 4
	model1 = acemodel(elements = [:Pb,:Te],
	        order = 2,
	        totaldegree = 5,
			r0 = (rnn(:Pb)+rnn(:Te))/2,
	        rcut = 6.0,
			#transform = (:agnesi, p, q),
			pair_transform = (:agnesi, p, q),
			pair_envelope = (:r, 2, 2),
			#envelope = (:x, 2, 2),
	        #Eref = [:Si => 0.0, :O => 0.0]
	)
	@show length(model1.basis);
end


n = 3
train_data = data[1:n]
test_data = data[n+1:497]

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

    plot([-6,-3], [-6,-3]; lc=:black, label="")

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
             title = @sprintf("PbTe Structures Used In Training:  %i out of %i\n", length(train_dataset), length(test_data)) *
                     @sprintf("RMSE (MAE) For Entire Dataset:  %.0f (%.0f) meV/atom", 1000*rmse, 1000*mae),
             titlefontsize = 8,
             yerror = model_std,
             xlabel="Energy [eV/atom]", xlims=(-3.7611,-3.7602),
             ylabel="Model Energy [eV/atom]", ylims=(-3.7611,-3.7602),
             aspect_ratio = :equal, color="#F0BB62")

    model_energies = [energy(model.potential,atoms)/length(atoms) for atoms in train_dataset]
    scatter!(extract_energies(train_dataset), model_energies;
             label="training set", color="#064635")
savefig("fitting_PbTe.png") 
end;

assess_model(model1, train_data)