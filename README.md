# MAGMA_PT
A collection of pseudo-transient Julia codes to model magmatic processes.

All dependencies (MAT v0.10.2, Parallelstencil v0.5.6 and Plots v1.21.3) are included in the Project.toml.

If you do not have an NVidia GPU you can run the code by setting `USE_GPU = false`. For tips on optimal parallel performance see the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) package.  

In the current version of `MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl`, plastic yielding can be switched off by choosing a very large value for `Cm_R` (e.g. 1e9 Pa). An inviscid solution can be obtained by increasing `Adis_R` (e.g. 1 Pa s). Constant viscosity can be enabled by replacing the `compute_eta!()` kernels by the `const_eta!()` and `const_etac!()`. Constant viscosity in the matrix and in the inclusion can be defined by `eta_const_R` and `eta_inc_R`. 

## Running the reference model
Open Julia in the project directory, or navigate there using `cd("Path/To/Project")`. Then with the following commands you can dowload all dependencies
```julia
julia> ] 
pkg> activate .
pkg> instantiate
```
Press backspace to leave the package manager.

Before running the code you must provide a path to a directory where output figures and data will be saved. Open `MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl` in your favorite editor and change `PathToVisu` and `PathToData`.
Then you can run the code by the `include("MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl")` command.


The visualization included here serves monitoring reasons only, the output data contained in the `.mat` files can be post-processed in Matlab.

The development of these codes was funded by the European Research Council through Consolidator Grant #771143 (MAGMA).
