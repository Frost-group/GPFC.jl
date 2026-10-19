module GPFCJuLIPExt

using GPFC
using JuLIP: read_extxyz, get_data
import GPFC: Read_JuLIP_Atoms

include(joinpath(@__DIR__, "..", "src", "JulibAtoms.jl"))

end
