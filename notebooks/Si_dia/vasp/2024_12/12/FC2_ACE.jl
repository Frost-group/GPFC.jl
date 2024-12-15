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
	using Pkg
	Pkg.activate(".")
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
	Pkg.add("JuLIP")
	using KernelFunctions, ForwardDiff, Zygote
	using LinearAlgebra, Einsum, Statistics
	using JuLIP, CSV, DataFrames, DelimitedFiles
	using Plots, StatsBase
	using ProgressMeter
end;

begin
	using Pkg
	Pkg.activate(".")
	#Pkg.activate("C:/Users/Keerati/.julia/environments/v1.11")
	Pkg.Registry.add("General") 
	Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
	Pkg.add("ACEpotentials")
	using ACEpotentials
end;


bead_tiny_dataset = JuLIP.read_extxyz("C:/Users/Keerati/Documents/GitHub/GPFC.jl/notebooks/Si_dia/vasp/2024_12/12/d_Si.extxyz")