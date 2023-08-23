# MAGMA_PT
A collection of pseudo-transient Julia codes to model magmatic processes.


## Running the reference model
Open Julia in the project directory, or navigate there using `cd("Path/To/Project")`. Then with the following commands you can dowload all dependencies (listed in Project.toml)
```julia
julia> ] 
pkg> activate .
```
Press backspace to leave the package manager.

Before running the code you must provide a path to an already existing directory where output figures and data will be saved. Open `MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl` in your favorite editor (such as Visual Studio Code) and change `PathToVisu` and `PathToData`.
If you do not have an NVidia GPU you can run the code by setting `USE_GPU = false`. For tips on optimal parallel performance see the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) package.  

Next, you can run the reference model with (after having created a directory for the output):
```julia
julia> using MAGMA_PT
julia> include("MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl")
```

The visualization included here serves monitoring purposes only, the output data contained in the `.mat` files can be post-processed in Matlab.

In the current version of `MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl`, plastic yielding can be switched off by choosing a very large value for `Cm_R` (e.g. 1e9 Pa). An inviscid solution can be obtained by increasing `Adis_R` (e.g. 1 Pa s). Constant viscosity can be enabled by replacing the `compute_eta!()` kernels by the `const_eta!()` and `const_etac!()`. Constant viscosity in the matrix and in the inclusion can be defined by `eta_const_R` and `eta_inc_R`. 

## Publication
The scripts here form part of a submitted manuscript:
- Kiss, D., Moulas, E., Kaus, B. J., & Spang, A. (2023). Decompression and fracturing caused by magmatically induced thermal stresses. Journal of Geophysical Research: Solid Earth, 128(3), e2022JB025341.

## Funding
The development of these codes was funded by the European Research Council through Consolidator Grant #771143 (MAGMA).
