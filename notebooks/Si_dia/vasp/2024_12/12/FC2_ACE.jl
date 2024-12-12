begin
	# add and load general packages used in this notebook.
	using Pkg
	Pkg.activate(".")
	Pkg.add("LaTeXStrings")
	Pkg.add("MultivariateStats")
	Pkg.add("Plots")
	Pkg.add("Suppressor")
	using LaTeXStrings, MultivariateStats, Plots, Printf, Statistics, Suppressor
end;

begin
    Pkg.add("KernelFunctions")
    Pkg.add("ForwardDiff")
    Pkg.add("Zygote")
	Pkg.add("LinearAlgebra")
    Pkg.add("Einsum")
    Pkg.add("Statistics")
	Pkg.add("CSV")
    Pkg.add("DataFrames")
    Pkg.add("DelimitedFiles")
	Pkg.add("Plots")
    Pkg.add("StatsBase")
    Pkg.add("ProgressMeter")
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum, Statistics
	using CSV, DataFrames, DelimitedFiles
	using Plots, StatsBase
	using ProgressMeter
end

Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
Pkg.add("ACEpotentials")

using GPFC

