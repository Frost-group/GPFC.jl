using Pkg
# Uncomment the next line if installing Julia for the first time
# Pkg.Registry.add("General")
Pkg.activate(".")
Pkg.add(["LaTeXStrings", "MultivariateStats", "Plots", "PrettyTables",
         "Suppressor", "ExtXYZ", "Unitful", "Distributed", "AtomsCalculators",])
# Add the ACE registry, which stores the ACEpotentials package information
#Pkg.Registry.add("General") 
Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add("ACEpotentials")

using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful
using ACEpotentials
Pkg.instantiate()

data_file = "d_Si.extxyz"
data = ExtXYZ.load(data_file)
Si_tiny_dataset = data[1:10]
Si_dataset = data[11:509]

println("The tiny dataset has ", length(Si_tiny_dataset), " structures.")
println("The large dataset has ", length(Si_dataset), " structures.")


r_cut = 6.0u"Å"
rnn = 2.35

rdf_tiny = ACEpotentials.get_rdf(Si_tiny_dataset, r_cut; rescale = true)
plt_rdf_1 = histogram(rdf_tiny[(:Si, :Si)], bins=150, label = "rdf",
                      title="Si_tiny_dataset", titlefontsize=10,
                      xlabel = L"r[\AA]", ylabel = "RDF", yticks = [],
                      xlims=(1.5,6), size=(400,200), left_margin = 2Plots.mm)
vline!(rnn * [1.0, 1.633, 1.915, 2.3, 2.5], label = "r1, r2, ...", lw=3)

rdf = ACEpotentials.get_rdf(Si_dataset, r_cut; rescale = true);
plt_rdf_2 = histogram(rdf[(:Si, :Si)], bins=150, label = "rdf",
                      title="Si_dataset", titlefontsize=10,
                      xlabel = L"r[\AA]", ylabel = "RDF", yticks = [],
                      xlims=(1.5,6), size=(400,200), left_margin = 2Plots.mm)
vline!(rnn * [1.0, 1.633, 1.915, 2.3, 2.5], label = "r1, r2, ...", lw=3)

plot(plt_rdf_1, plt_rdf_2, layout=(2,1), size=(400,400))


function extract_energies(dataset)
    energies = []
    for atoms in dataset
        for key in keys(atoms)
            if lowercase(String(key)) == "energy"
                push!(energies, atoms[key] / length(atoms))
            end
        end
    end
    return energies
end;


atom = Si_dataset[1]


Si_dataset_energies = extract_energies(Si_dataset)
;  # the ; is just to suppress the ouput



model = ace1_model(elements = [:Si],
                   rcut = 5.5,
                   order = 3,        # body-order - 1
                   totaldegree = 8 );

descriptors = []
    for system in Si_tiny_dataset
        struct_descriptor = sum(site_descriptors(system, model)) / length(system)
        push!(descriptors, struct_descriptor)
    end 


function dRdY(model::ACEModel, 
        Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
        ps, st) where {T}

    i_z0 = _z2i(model.rbasis, Z0)

    if length(Rs) == 0 
        return model.Vref.E0[Z0], SVector{3, T}[] 
    end 

    @no_escape begin 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ---------- EMBEDDINGS ------------
        # (these are done in forward mode, so not part of the fwd, bwd passes)

        # get the radii 
        rs, ∇rs = @withalloc radii_ed!(Rs)

        # evaluate the radial basis
        Rnl, dRnl = @withalloc evaluate_ed_batched!(model.rbasis, rs, Z0, Zs, 
                                  ps.rbasis, st.rbasis)

         # evaluate the Y basis
        Ylm, dYlm = @withalloc P4ML.evaluate_ed!(model.ybasis, Rs)

        # Forward Pass through the tensor 
        # keep intermediates to be used in backward pass 
        B, intermediates = @withalloc evaluate!(model.tensor, Rnl, Ylm)

        # contract with params 
        # (here we can insert another nonlinearity instead of the simple dot)
        Ei = dot(B, (@view ps.WB[:, i_z0]))

        # Start the backward pass 
        # ∂Ei / ∂B = WB[i_z0]
        ∂B = @view ps.WB[:, i_z0]

        # backward pass through tensor 
        ∂Rnl, ∂Ylm = @withalloc pullback!(∂B, model.tensor, Rnl, Ylm, intermediates)

        return Rnl, ∂Rnl, Ylm, ∂Ylm
    end
end