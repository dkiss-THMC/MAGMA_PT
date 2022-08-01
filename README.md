# MAGMA_PT
A collection of pseudo-transient Julia codes to model magmatic processes.

All dependencies (MAT v0.10.2, Pallelstencil v0.5.6 and Plots v1.21.3) are included in the Project.toml.

If you do not have an NVidia GPU you can run the code by setting USE_GPU = false. For tips on optimal parallel performance see the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) package.  

In the current version of `MAGMA_2D_PT_VEPCoolingPlasticIntrusion.jl`, plastic yielding can be switched off by choosing a very large value for Cm_R (e.g. 1e9 Pa). An inviscid solution can be obtained by increasing Adis_R (1). Constant viscosity can be enabled by replacing the `compute_eta!()` kernels by the const_eta!() and cons_etac()!. Constant viscosity in the matrix (e.g. 1 Pa s) and in the inclusion can be defined by `eta_const_R` and `eta_inc_R`. 

## Running the reference model
```julia
julia> ]
pkg> activate .
```


The visualization included here serves monitoring reasons only, the output data contained in the `.mat` files can be post-processed in Matlab.

The development of these codes was funded by the European Research Council through Consolidator Grant #771143 (MAGMA).
