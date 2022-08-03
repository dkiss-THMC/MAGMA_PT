const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT

##################################################
@views function MAGMA_2D_PT_VEPCoolingPlasticIntrusion()
    ######################################
    ###### User's input starts here ######
    ######################################
    # Physics with dimensions (_R stands for real values)
    Lx_R      = 10.5e3                          # Width of the domain, m
    Ymax_R    = 2e3                             # Maximum Y coordinate, m
    Ly_R      = 12.5e3                          # Height of the domain, m
    Amp_R     = 0.0                             # Amplitude of topography, m
    dV_R      = 0.0                             # velocity difference, for initial loading, m/s
    εbg_R     = dV_R/Lx_R                       # Background strain rate, 1/s
    gval_R    = -9.81                           # gravity acceleration ( hard coded for y direction), m/s^2
    β_R       = 1e-11                           # compressibility, 1/Pa
    α_R       = 3e-5                            # thermal expansion coefficient, 1/K
    ρRock_R   = 2650.0                          # reference density at (P=0 atm, and T=0 °C), kg/m^3
    ρAir_R    = 1.225                           # reference density at (P=0 atm, and T=0 °C), kg/m^3
    Tref_R    = 273.15                          # reference temperature, K
    Ttop_R    = 10+273.15                       # temperature on top, K   
    Tbottom_R = 450+273.15                      # temperature on the bottom, K
    Tmagma_R  = 750+273.15                      # initial temperature of the magma chamber, K
    Cp_R      = 1050.0                          # heat capacity at const. P, J/kg/K
    λ_R       = 3.0                             # thermal conductivity, W/m/K
    Qr_R      = 1e-6                            # volumetric radiogenic heat production, W/m^3
    Adis_R    = 1.67e-24                        # pre-exponential factor for rheology in invariant form, Pa^(-n) s^(-1) 
    ndis_R    = 3.3                             # power-law exponent, []
    Edis_R    = 187e3                           # activation energy, J/mol 
    R_R       = 8.3145                          # universal gas constant, J/mol/K
    η_min_R   = 1e19                            # minimum viscosity, Pa s
    η_max_R   = 1e23                            # maximum viscosity, Pa s
    F_R       = 1.0                             # geometrical coefficient for flow law, []
    ν         = 0.25                            # Poissions's ratio
    μ_R       = 3.0/β_R*(1.0-2.0*ν)/2.0/(1.0+ν) # elastic shear modulus, Pa
    φ_R       = 30.0                            # friction angle, deg
    ψ_R       = 15.0                            # dilation angle, deg
    Cm_R      = 15e6                            # mean cohesion, Pa
    C_amp_R   = 0.2*Cm_R                        # amplitude of max cohesion perturbation, Pa
    σ_T_to_C  = 0.5                             # σ_T/C, the ratio pf cohesion and tensile strength, []
    δσ_T_R    = 1e5                             # tensile strength - minimum mean stress, Pa
    η_const_R = 1e30                            # constant viscosity, Pa s
    η_inc_R   = 1e30                            # constant viscosity, Pa s
    tt_R      = 3.25e5*(365.25*24*3600)         # total simulation time (after last pulse), y
    Soft_R    = Cm_R*0                          # Cohesion softening parameter, dC/dgamma, Pa
    C_min_R   = 1e6                             # minimum cohesion, Pa
    # Intrusion parameters, extend the arrays to add more intrusion events
    xO_R = [ 0.0   -1e3    1e3     0.0  ]                    # x coordinate of center, m
    yO_R = [-5.0e3 -5e3   -4.5e3  -5.5e3]                    # y coordinate of center, m
    rx_R = [ 1.5e3  1.0e3  0.5e3   1.5e3]                    # horizontal semi axis, m
    ry_R = [ 1.5e3  0.5e3  1.0e3   1.0e3]                    # vertical semi axis, m
    px_R = [ 2.0    2.0    2.0     2.0  ]                    # power of the x term, []
    py_R = [ 2.0    2.0    2.0     2.0  ]                    # power of the y term, []
    tI_R = [ 0.0    2.0    5e3     1e4  ]*(365.25*24*3600)   # time of injection (must start with 0.0), s
    # independent scales
    ηsc       = 1e23                           # characteristic viscosity
    Ksc       = λ_R/ρRock_R/Cp_R               # characteristic viscosity
    Lsc       = 1e3
    Tsc       = (Tmagma_R-Ttop_R)/2.0
    #######################################
    ####### User's input pauses here ######
    ###### Please keep scrooling down #####
    #######################################
    # dependent scales
    tsc       = Lsc^2.0/Ksc
    εsc       = 1.0/tsc
    Vsc       = Lsc/tsc
    gsc       = Lsc/tsc^2.0
    Psc       = ηsc/tsc
    βsc       = 1.0/Psc
    ρsc       = Psc/gsc/Lsc
    αsc       = 1.0/Tsc
    λsc       = Psc*Lsc^2.0/tsc/Tsc
    Cpsc      = λsc/Ksc/ρsc
    Qsc       = ρsc*Cpsc*Tsc/tsc
    Adissc    = Psc^(-ndis_R)*1.0/tsc
    Esc       = Psc*Lsc^3.0
    Rsc       = Psc*Lsc^3.0/Tsc
    # Nondimensional Physics
    Lx        = Lx_R/Lsc
    Ly        = Ly_R/Lsc
    Ymax      = Ymax_R/Lsc
    Amp       = Amp_R/Lsc
    dV        = dV_R/Vsc
    εbg       = εbg_R/εsc
    gval      = gval_R/gsc
    β         = β_R/βsc
    α         = α_R/αsc
    ρRock     = ρRock_R/ρsc
    ρAir      = ρAir_R/ρsc
    Tref      = Tref_R/Tsc
    Ttop      = Ttop_R/Tsc
    Tbottom   = Tbottom_R/Tsc
    Tmagma    = Tmagma_R/Tsc
    Cp        = Cp_R/Cpsc
    λ         = λ_R/λsc
    Qr        = Qr_R/Qsc
    Adis      = Adis_R/Adissc
    ndis      = ndis_R
    Edis      = Edis_R/Esc
    R         = R_R/Rsc
    η_min     = η_min_R/ηsc
    η_max     = η_max_R/ηsc
    η_const   = η_const_R/ηsc
    η_inc     = η_inc_R/ηsc
    F         = F_R
    μ         = μ_R/Psc
    φ         = φ_R 
    ψ         = ψ_R
    Cm        = Cm_R/Psc
    C_amp     = C_amp_R/Psc
    δσ_T      = δσ_T_R/Psc
    Soft      = Soft_R./Psc
    C_min     = C_min_R./Psc
    tt        = tt_R/tsc
    xO        = xO_R./Lsc 
    yO        = yO_R./Lsc
    rx        = rx_R./Lsc 
    ry        = ry_R./Lsc 
    px        = px_R
    py        = py_R
    tI        = tI_R./tsc
    ######################################
    ###### User's input resumes here #####
    ######################################
    # Numerics
    nx        = 16*16 - 1       # number of cells in x; should be a (mulitple of 16)-1 for optimal GPU perf
    ny        = 19*16 - 1       # number of cells in y; should be a (mulitple of 16)-1 for optimal GPU perf
    nt        = 512             # number of physical timesteps
    nout      = 1000            # print residuals after each n iterations
    nData     = 4               # write output data each after n timesteps
    tolΔV     = 1e-17/εsc       # tolerance of volumetric strain rate, to scale ResP [ΔV] 
    tolΔσ     = 1e3/Psc         # tolerance of stress gradient,        to scale RVx and RVy [ΔP/dy]
    tolΔT     = 1e-3/Tsc        # tolerance of temp. change,           to scale ResT [ΔT/dt]
    iterMin   = 25              # minimum number of iterations
    iterMax   = 5e4             # maximum number of iterations
    βn        = 0.5             # numerical compressibility, for the incompressible solver
    ξ_ic      = 2.5             # damping parameter for incompressible solver
    dampX_ic  = 1.0-ξ_ic/nx     # damping in x direction, for incompressible solver
    dampY_ic  = 1.0-ξ_ic/ny     # damping in y direction, for incompressible solver
    CLFV_ic   = 1.0/2.0         # PT timestep modifier, play with these if it explodes
    CLFP_ic   = 1.0/2.0         # PT timestep modifier, play with these if it explodes
    ξ_c       = 3.0             # damping parameter for compressible solver
    dampX_c   = 1.0-ξ_c/nx      # damping in x direction, for compressible solver
    dampY_c   = 1.0-ξ_c/ny      # damping in y direction, for compressible solver
    dampP_c   = 0.4             # damping of P, for compressible solver
    dampT_c   = 0.0             # damping of T, for compressible solver
    CLFV_c    = 1.0/5.0         # PT timestep modifier, play with these if it explodes
    CLFP_c    = 1.0/4.0         # PT timestep modifier, play with these if it explodes
    CLFT      = 0.25            # PT timestep modifier, play with these if it explodes
    relη      = 0.05            # relaxation factor for power-law viscosity
    relpl     = 0.5             # Duvaut-Lyons relaxation factor for plasticity
    η_vpl     = 0.0/ηsc         # viscosity for Prezyna's regularization
    ndiff     = 8               # number of smoothing steps applied to the cohesion field
    nTini     = 200             # number of smoothing steps applied on the initial T field 
    # Path
    ENV["GKSwstype"]="nul"; 
    PathToVisu = "/EnterGlobalPathToExistingDirectory/";  # Enter global path to the directory where visualisation is to be stored
    #if isdir(PathToVisu)==false mkdir(PathToVisu) end;      # create path directory if it does not exist, convienient but optional, as it does not work on all systems
    anim = Animation(PathToVisu,String[])
    PathToData = "/EnterGlobalPathToExistingDirectory/";  # Enter global path to the directory where output data is to be stored
    #if isdir(PathToData)==false mkdir(PathToData) end;      # create path directory if it does not exist, convienient but optional, as it does not work on all systems
    println("Animation directory: $(anim.dir)")
    
    ###################################
    ###### User's input ends here #####
    ###################################

    # Derived numerics
    dx       = Lx/nx   
    dy       = Ly/(ny-0.5)
    min_dxy2 = min(dx,dy)^2.0
    max_nxy  = max(nx,ny)
    ΔT       = (Ttop-Tbottom)/Ly
    # Points defining the piecewisely linear yield function for mean cohesion
    σ_Tm     = σ_T_to_C *Cm
    X0σ      = σ_Tm - δσ_T                        # σm at the intersection of cutoff with τII = 0 (with the horizontal coordinate axis)
    X0τ      = 0.0                                # τII at the intersection of cutoff with τII = 0
    X1σ      = σ_Tm - δσ_T                        # σm at the intersection of cutoff and Mode-1
    X1τ      = -X1σ + σ_Tm                        # τII at the intersection of cutoff and Mode-1
    X2σ      = (σ_Tm-Cm*cosd(φ))/(1.0-sind(φ))    # σm at the intersection of Drucker-Prager and Mode-1
    X2τ      = -X2σ + σ_Tm                        # τII at the intersection of Drucker-Prager and Mode-1
    
    # Array allocations
    P        =  zeros(nx  ,ny  )
    P_tr     = @zeros(nx  ,ny  )
    P_old    = @zeros(nx  ,ny  )
    P_old_Lag= @zeros(nx  ,ny  )
    T        =  zeros(nx  ,ny  )
    T_old    = @zeros(nx  ,ny  )
    T_old_Lag= @zeros(nx  ,ny  )
    qx       = @zeros(nx+1,ny  )
    qy       = @zeros(nx  ,ny+1)
    ρ        =  zeros(nx  ,ny  )
    ρ_old    = @zeros(nx  ,ny  )
    ρ_old_Lag= @zeros(nx  ,ny  )
    C_old    = @zeros(nx  ,ny  )
    C_old_Lag= @zeros(nx  ,ny  )
    η        = @zeros(nx  ,ny  )
    ηc       = @zeros(nx+1,ny+1)
    η_ve     = @zeros(nx  ,ny  )
    ηc_ve    = @zeros(nx+1,ny+1)  
    η_vep    = @zeros(nx  ,ny  )
    ηc_vep   = @zeros(nx+1,ny+1)
    η_old    = @zeros(nx  ,ny  )     
    dτP      = @zeros(nx  ,ny  )
    ∇V       = @zeros(nx  ,ny  )
    Vx       = @zeros(nx+1,ny  )
    Vy       = @zeros(nx  ,ny+1)
    σxx      = @zeros(nx  ,ny  )
    σyy      = @zeros(nx  ,ny  )
    σzz      = @zeros(nx  ,ny  )
    σxy      = @zeros(nx+1,ny+1)
    σxx_tr   = @zeros(nx  ,ny  )
    σyy_tr   = @zeros(nx  ,ny  )
    σzz_tr   = @zeros(nx  ,ny  )
    σxy_tr   = @zeros(nx+1,ny+1)
    σxx_old  = @zeros(nx  ,ny  )
    σyy_old  = @zeros(nx  ,ny  )
    σzz_old  = @zeros(nx  ,ny  )
    σxy_old  = @zeros(nx+1,ny+1)
    σxx_old_Lag= @zeros(nx  ,ny  )
    σyy_old_Lag= @zeros(nx  ,ny  )
    σzz_old_Lag= @zeros(nx  ,ny  )
    σxy_old_Lag= @zeros(nx+1,ny+1)
    τII      = @zeros(nx  ,ny  )
    τIIc     = @zeros(nx-1,ny-1)
    τII_tr   = @zeros(nx  ,ny  )
    τIIc_tr  = @zeros(nx-1,ny-1)
    P_tr     = @zeros(nx  ,ny  )
    F_tr     = @zeros(nx  ,ny  )
    ResτII   = @zeros(nx  ,ny  )
    εxx      = @zeros(nx  ,ny  )
    εyy      = @zeros(nx  ,ny  )
    εzz      = @zeros(nx  ,ny  )
    εxy      = @zeros(nx+1,ny+1)
    εxx_ve   = @zeros(nx  ,ny  )
    εyy_ve   = @zeros(nx  ,ny  )
    εzz_ve   = @zeros(nx  ,ny  )
    εxy_ve   = @zeros(nx+1,ny+1)
    ε_vol_pl = @zeros(nx  ,ny  )
    εII_pl   = @zeros(nx  ,ny  )
    domain_pl= @zeros(nx  ,ny  )
    ε_vol_pl_old = @zeros(nx  ,ny  )
    εII_pl_old   = @zeros(nx  ,ny  )
    γ_vol_pl = @zeros(nx  ,ny  )
    γII_pl   = @zeros(nx  ,ny  )
    εII      = @zeros(nx  ,ny  )
    εII_ve   = @zeros(nx  ,ny  )
    εIIc_ve  = @zeros(nx-1,ny-1)
    εIIc     = @zeros(nx-1,ny-1)
    εII_vis  = @zeros(nx  ,ny  )
    εIIc_vis = @zeros(nx-1,ny-1)
    SH       = @zeros(nx  ,ny  )
    ResP     = @zeros(nx  ,ny  )
    dPdτ     = @zeros(nx  ,ny  )
    RVx      = @zeros(nx-1,ny  )
    RVy      = @zeros(nx  ,ny-1)
    dTdt     = @zeros(nx  ,ny  )
    dTdτ     = @zeros(nx  ,ny  )
    ResT     = @zeros(nx  ,ny  )
    LinT     = @zeros(nx  ,ny  )
    dVxdτ    = @zeros(nx-1,ny  )
    dVydτ    = @zeros(nx  ,ny-1)
    dτVx     = @zeros(nx-1,ny  )
    dτVy     = @zeros(nx  ,ny-1)
    dτT      = @zeros(nx  ,ny  )
    σ_T      = @zeros(nx  ,ny  )
    x1σ      = @zeros(nx  ,ny  )
    x1τ      = @zeros(nx  ,ny  )
    x2σ      = @zeros(nx  ,ny  )
    x2τ      = @zeros(nx  ,ny  )
    SAir     =  zeros(nx  ,ny  )
    SAir_old = @zeros(nx  ,ny  )
    SAirv    =  zeros(nx+1,ny+1)
    ρref     =  zeros(nx  ,ny  )

    # Grid coordinates
    x        = [-Lx/2.0+dx/2.0    + (ix-1)*dx for ix = 1:nx  ]
    y        = [-(Ly-Ymax)+dy/2.0 + (iy-1)*dy for iy = 1:ny  ]
    xv       = [-Lx/2.0           + (ix-1)*dx for ix = 1:nx+1]
    yv       = [-(Ly-Ymax)        + (iy-1)*dy for iy = 1:ny+1]
    xc       = [x[ix]    for ix=1:nx  , iy=1:ny  ]
    yc       = [y[iy]    for ix=1:nx  , iy=1:ny  ]
    xVx      = [xv[ix]   for ix=1:nx+1, iy=1:ny  ]
    yVx      = [y[iy]    for ix=1:nx+1, iy=1:ny  ]
    xVy      = [x[ix]    for ix=1:nx  , iy=1:ny+1]
    yVy      = [yv[iy]   for ix=1:nx  , iy=1:ny+1]
    xσxy     = [xv[ix]   for ix=1:nx+1, iy=1:ny+1]
    yσxy     = [yv[iy]   for ix=1:nx+1, iy=1:ny+1]
    Xmin     = Data.Number(minimum(xc))
    Ymin     = Data.Number(minimum(yc))

    # Initial conditions
    SAir        .= ( Amp.*cos.(pi.*xc./Lx) .- (yc.-dy./2.0) )./dy                                       # free surface minus lower cell edge coordinate
    SAir        .= (SAir .> 0.0).*(SAir .< 1.0).*SAir + (SAir .> 1.0)                                   # 0.0 if SAir<0.0 and 1.0 if SAir>1.0, 100% Sticky Air = 0.0, and 0% Sticky Air = 1.0
    SAirv       .= ( Amp.*cos.(pi.*xσxy./Lx) .- (yσxy.-dy./2.0) )./dy                                   # free surface minus lower cell center coordinate
    SAirv       .= (SAirv .> 0.0).*(SAirv .< 1.0).*SAirv .+ (SAirv .> 1.0)                              # 0.0 if SAirv<0.0 and 1.0 if SAirv>1.0, 100% Sticky Air = 0.0, and 0% Sticky Air = 1.0
    P           .= -reverse(cumsum(ρRock.*gval.*reverse(SAir,dims=2)*dy,dims=2),dims=2)
    T           .= Ttop .+ reverse(cumsum(-ΔT.*reverse(SAir,dims=2)*dy,dims=2),dims=2)
    ρref        .= ρRock.*SAir .+ ρAir.*(1.0.-SAir)
    ρ           .= ρref .* exp.(β.*P .- α.*(T.-Tref) )
    P            = Data.Array(P)
    T            = Data.Array(T)
    ρ            = Data.Array(ρ)
    ρref         = Data.Array(ρref)
    SAir         = Data.Array(SAir)
    SAirv        = Data.Array(SAirv)
    xc_old       = xc
    yc_old       = yc
    xc           = Data.Array(xc)
    yc           = Data.Array(yc)
    xc_old       = Data.Array(xc_old)
    yc_old       = Data.Array(yc_old)
    dt           = min_dxy2/(λ/maximum(ρ)/Cp)/4.5
    for it = 1:25e3
        @parallel compute_heatflux!(T, qx, qy, λ, dx, dy)    
        @parallel InitialT!(dTdt,T, qx, qy, ρ, Cp, Qr, dt, dx, dy, SAir)
        @parallel (1:nx) Bottom_T!(qy, T, Tbottom, λ, dy)  
    end
    event        = 1
    rad          = IntrusionCoordinates(Lx,dx,nx,Ly,dy,ny,Ymax,6,xO,yO,rx,ry,px,py,event)
    @parallel TAnomaly!(T, Data.Array(rad), Tmagma )   
    for it = 1:nTini
        @parallel compute_heatflux!(T, qx, qy, λ, dx, dy)    
        @parallel InitialT!(dTdt,T, qx, qy, ρ, Cp, Qr, dt, dx, dy, SAir)
        @parallel (1:nx) Bottom_T!(qy, T, Tbottom, λ, dy)
    end
    @parallel compute_ρ!(ρ, P, T, ρref, Tref, α, β) 
    #@parallel const_η!(η, Data.Array(rad), η_const, η_inc) 
    #@parallel const_ηc!(η, ηc) 
    
    # Initalize cohesion with random noise
    Co            = rand(nx+ndiff,ny+ndiff)
    for it = 1:ndiff
        Co[2:end-1,2:end-1] = Co[2:end-1,2:end-1] + 1.0/8.1*(diff(diff(Co[:,2:end-1],dims=1),dims=1) + diff(diff(Co[2:end-1,:],dims=2),dims=2))
    end
    C           = Co[1+Int(ndiff/2):end-Int(ndiff/2),1+Int(ndiff/2):end-Int(ndiff/2)]
    C          .= Cm .+ C_amp.*((C.-minimum(C))./(maximum(C)-minimum(C)).-0.5)
    C           = Data.Array(C)
    @parallel set_plastic_parameters!(C, σ_T, x1σ, x1τ, x2σ, x2τ, δσ_T, φ, ψ, σ_T_to_C)
    p1 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(P*Psc/1e6)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:inferno, title="Pressure [MPa]")
    p2 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(T*Tsc.-273.15)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", ylabel="y [km]", c=:inferno, title="Temperature [°C]")
    p3 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(ρ*ρsc)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", c=:inferno, title="Density [kg/m³]")
    p4 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(C*Psc/1e6)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:inferno, title="Cohesion [MPa]")
    p5 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(SAir)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:inferno, title="Cohesion [MPa]")
    p5 = plot!( xv*Lsc/1e3,(Amp.*cos.(pi.*xv./Lx))*Lsc/1e3 )
    plot(p1, p2, p3, p4)
    display(plot!(size=(1600,800), margin = 10Plots.mm))
    @printf("α/β = %1.3e, β/dt = %1.3e \n", α_R/β_R, β_R/(dt*tsc))

    # Initial incompressible solution
    err=2.0; iter=1; err_evo1=[]; err_evo2=[]
    while err > 1.0 && iter <= iterMax
        @parallel compute_∇V!( ∇V, Vx, Vy, dx, dy)
        @parallel compute_εij!( ∇V, εxx, εyy, εzz, εxy, Vx, Vy, dx, dy)
        @parallel compute_εII!(εxx, εyy, εzz, εxy, εII,εIIc)
        if iter == 1  releta = 1.0 else releta = relη end
        @parallel compute_η!(η, ηc, T, εII, releta, Adis, ndis, Edis, R)
        @parallel η_cutoff!(η, ηc, η_min, η_max, SAir, SAirv)
        @parallel compute_timesteps_ic!(dτVx, dτVy, dτP, η, ηc, CLFV_ic, CLFP_ic, βn, min_dxy2, Data.Number(max_nxy))
        @parallel compute_P_ic!(∇V, P, dτP)
        @parallel compute_ρ!(ρ, P, T, ρref, Tref, α, β)
        @parallel compute_σ!(P, σxx, σyy, σzz, σxy, εxx, εyy, εzz, εxy,  η, ηc, ∇V, βn)
        @parallel (1:nx  ) FreeSurface_σyy!(σyy)
        @parallel (1:nx+1) FreeSurface_σxy!(σxy)
        @parallel compute_dV!(RVx, RVy, dVxdτ, dVydτ, ρ, σxx, σyy, σxy, dampX_ic, dampY_ic, gval, dx, dy)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
        @parallel (1:nx  ) FreeSurface_Vy!(Vy, P, η, Vx, dx, dy)
        if mod(iter,1000)==0
            global max_RVx, max_RVy, max_∇V
            max_RVx = maximum(abs.(RVx)); max_RVy = maximum(abs.(RVy)); max_∇V = maximum(abs.(-∇V))
            err = maximum([max_RVx, max_RVy, max_∇V])
            push!(err_evo1, maximum([max_RVx, max_RVy, max_∇V])); push!(err_evo2,iter)
            @printf("Iterations = %d, err = %1.3e [max_RVx=%1.3e, max_RVy=%1.3e, max_∇V=%1.3e] \n", iter, err, max_RVx, max_RVy, max_∇V)
        end
        iter+=1
    end
    @parallel CopyParallelArray!(P, P_old_Lag)
    @parallel CopyParallelArray!(T, T_old_Lag)
    @parallel CopyParallelArray!(σxx , σxx_old_Lag)
    @parallel CopyParallelArray!(σyy , σyy_old_Lag)
    @parallel CopyParallelArray!(σzz , σzz_old_Lag)
    @parallel CopyParallelArray!(σxy , σxy_old_Lag)
    P_ini = Array(P)
    T_ini = Array(T)
    ρ_ini = Array(ρ)
    @parallel CopyParallelArray!(P, P_tr)
    @parallel CopyParallelArray!(εII, εII_vis)
    @parallel CopyParallelArray!(εIIc, εIIc_vis)
    @parallel CopyParallelArray!(η, η_vep)
    p1 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(P*Psc/1e6)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3),clims=(0,0.1), ylabel="y [km]", c=:inferno, title="Pressure [MPa]")
    p2 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(T*Tsc.-273.15)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", ylabel="y [km]", c=:inferno, title="Temperature [°C]")
    p3 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(ρ*ρsc)', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", c=:inferno, title="Density [kg/m³]")
    p4 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(log10.(η*ηsc))', aspect_ratio=1, xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", c=:inferno, title="Effective viscosity[Pa.s]")
    plot(p1, p2, p3, p4)
    display(plot!(size=(1600,800), margin = 10Plots.mm))

    # Time loop
    @parallel set_far_field!(Vx, εbg, Data.Array(xVx))
    βn    = 0.0
    time  = 0.0; event = 2; itref = 0; niter=0; it = 0
    while it-itref <= nt
        it    = it + 1
        dt    = tt/nt^2.0*((it-itref)^2.0-((it-itref)-1.0)^2.0);
        if event <= length(tI)
            if time+dt > tI[event]
                dt    = tI[event]-time;
            end
            if  time == tI[event]
                itref = it-1
                dt    = tt/nt^2.0*((it-itref)^2.0-((it-itref)-1.0)^2.0);
                rad   = IntrusionCoordinates(Lx,dx,nx,Ly,dy,ny,Ymax,6,xO,yO,rx,ry,px,py,event)
                @parallel TAnomaly!(T, Data.Array(rad), Tmagma )
                event = event+1
            end
        end
        time  = time + dt;
        η_min = μ*dt/1e3        # minimum viscosity, Pa s
        η_max = μ*dt*1e2        # maximum viscosity, Pa s
        @parallel CopyParallelArray!(SAir, SAir_old)
        @parallel CopyParallelArray!(P   , P_old  )
        @parallel CopyParallelArray!(T   , T_old  )
        @parallel CopyParallelArray!(σxx , σxx_old)
        @parallel CopyParallelArray!(σyy , σyy_old)
        @parallel CopyParallelArray!(σzz , σzz_old)
        @parallel CopyParallelArray!(σxy , σxy_old)
        @parallel CopyParallelArray!(ρ   , ρ_old  )
        @parallel CopyParallelArray!(C   , C_old  )
        err=2.0; iter=1; err_evo1=[]; err_evo2=[]; 
        while err > 1.0 && iter <= iterMax
            if (niter==11)  global wtime0 = Base.time()  end
            @parallel compute_∇V!( ∇V, Vx, Vy, dx, dy)
            @parallel compute_ResP!(∇V, ResP, dPdτ, P_tr, P_old_Lag, T, T_old_Lag, Data.Number(β), Data.Number(α), Data.Number(dampP_c), Data.Number(dt))      
            @parallel Update_P!(dPdτ, P_tr, dτP)
            @parallel compute_εij!( ∇V, εxx, εyy, εzz, εxy, Vx, Vy, dx, dy)
            if iter == 1  releta = 1.0 else releta = relη end
            @parallel compute_η!(η, ηc, T, εII_vis, releta, Adis, ndis, Edis, R)
            @parallel compute_η_ve!(η_ve, ηc_ve, η, ηc, dt, μ)
            #@parallel η_cutoff!(η_ve, ηc_ve, η_min, η_max, SAir, SAirv)
            @parallel compute_εij_ve!(εxx_ve, εyy_ve, εzz_ve, εxy_ve, εxx, εyy, εzz, εxy,σxx_old_Lag, σyy_old_Lag, σzz_old_Lag, σxy_old_Lag, P_old_Lag, μ, dt) 
            @parallel compute_εII!(εxx_ve, εyy_ve, εzz_ve, εxy_ve, εII_ve, εIIc_ve)
            @parallel compute_σ!(P_tr, σxx_tr, σyy_tr, σzz_tr, σxy_tr, εxx_ve, εyy_ve, εzz_ve, εxy_ve, η_ve, ηc_ve, ∇V, βn)
            @parallel (1:nx  ) FreeSurface_σyy!(σyy_tr)
            @parallel (1:nx+1) FreeSurface_σxy!(σxy_tr)
            @parallel compute_τII_tr!(τII_tr, η_ve, εII_ve, τIIc_tr, ηc_ve, εIIc_ve)
            @parallel CopyParallelArray!(εII_pl, εII_pl_old)
            @parallel CopyParallelArray!(ε_vol_pl, ε_vol_pl_old)
            if (it==itref+1 && iter>1000) || (it-itref)!=1
            @parallel (1:nx,1:ny) update_plastic_corrections!(F_tr, τII_tr, P_tr, εII_pl, ε_vol_pl, εII_pl_old, ε_vol_pl_old, domain_pl, η_ve, η_vpl, dt, β, φ, ψ, C, σ_T, δσ_T, x1σ, x1τ, x2σ, x2τ, relpl)
            end
            @parallel plastic_stress_corrections!(σxx, σyy, σzz, σxy, σxx_tr, σyy_tr, σzz_tr, σxy_tr, τII, τIIc, τII_tr, τIIc_tr, P_tr, P, εII_pl, ε_vol_pl, η_ve, ηc_ve, dt, β)
            @parallel (1:nx  ) FreeSurface_σyy!(σyy)
            @parallel (1:nx+1) FreeSurface_σxy!(σxy)
            @parallel CalculateVEP_viscosity!(τII, εII_ve, η_vep, τIIc, ηc_vep)
            @parallel compute_εII_vis!(εII_vis, η, τII)
            @parallel compute_timesteps_c!(dτVx, dτVy, dτP, dτT, η_vep, ηc_vep, ρRock, dt, λ, Cp, CLFV_c, CLFP_c, CLFT, min_dxy2, Data.Number(max_nxy))
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),ρ_old_Lag,ρ_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),SAir,SAir_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),P_old_Lag,P_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),T_old_Lag,T_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),σxx_old_Lag,σxx_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),σyy_old_Lag,σyy_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),σzz_old_Lag,σzz_old,xc_old,yc_old)
            @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),σxy_old_Lag,σxy_old,xc_old,yc_old)
            @parallel SAirv!(SAirv, SAir)
            @parallel update_ρ!(ρ,ρ_old_Lag, ∇V, dt)
            @parallel compute_dV!(RVx, RVy, dVxdτ, dVydτ, ρ, σxx, σyy, σxy, dampX_c, dampY_c, gval, dx, dy)
            @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
            @parallel (1:nx  ) FreeSurface_Vy_vep!(Vy, P, σyy_old_Lag, P_old_Lag, η_ve, Vx, εII_pl, μ, dt, dx, dy)
            @parallel compute_OldPos!(xc_old, yc_old, xc, yc, Vx, Vy, dt)
            if iter == 1 || mod(iter,1) ==0
            @parallel compute_heatflux!(T, qx, qy, λ, dx, dy)
            @parallel compute_SH!(SH, τII, εII_vis, εII_pl, P, ε_vol_pl)
            @parallel compute_T!(dTdt,ResT,T, T_old_Lag, P, P_old_Lag, qx, qy, ρ, SH, dTdτ, dτT, α, Cp, Qr, dt, dx, dy, dampT_c, SAir)
            @parallel (1:nx) Surface_T!(T, Ttop)
            @parallel (1:nx) Bottom_T!(qy, T, Tbottom, λ, dy)
            end
            if mod(iter,nout)==0
                @parallel compute_heatflux!(T_old, qx, qy, λ, dx, dy)
                @parallel LinearTermsInT!(LinT,T, T_old, qx, qy, ρ, Cp, Qr, dt, dx, dy) 
                @parallel StrainRateResidual!(ResτII, εII_ve, εII_pl, η_ve, τII, η, dt, μ) 
                global max_RVx, max_RVy, max_ResP, max_ResT, max_ResτII
                max_RVx = maximum(abs.(RVx))/(tolΔσ/dx); max_RVy = maximum(abs.(RVy))/(tolΔσ/dy); max_ResP = maximum(abs.(ResP))/tolΔV; max_ResT = maximum(abs.(ResT))/(tolΔT/dt); max_ResτII = maximum(abs.(ResτII)); 
                err = maximum([max_RVx, max_RVy, max_ResP, max_ResT])
                push!(err_evo1, maximum([max_RVx, max_RVy,  max_ResP, max_ResT])); push!(err_evo2,iter)
                @printf("Time step =%d, Iterations = %d, err = %1.3e [max_RVx=%1.3e, max_RVy=%1.3e, max_ResP=%1.3e, max_ResT=%1.3e, max_ResτII=%1.3e] \n", it, iter, err, max_RVx, max_RVy, max_ResP, max_ResT, max_ResτII)
                @printf("η_eff_elastic = %1.3e, dt/β = %1.3e] \n", μ_R*dt*tsc , dt/β*ηsc)
            end
            iter+=1; niter+=1
        end
        @parallel (1:nx,1:ny) interp2!(Xmin,Data.Number(nx),Data.Number(dx),Ymin,Data.Number(ny),Data.Number(dy),C,C,xc_old,yc_old)
        @parallel set_plastic_parameters!(C, σ_T, x1σ, x1τ, x2σ, x2τ, δσ_T, φ, ψ, σ_T_to_C)
        @parallel compute_εII!(εxx, εyy, εzz, εxy, εII, εIIc)
        @parallel TimeIntegration(γ_vol_pl,ε_vol_pl, dt)
        @parallel TimeIntegration(γII_pl  ,εII_pl  , dt)

    # Visualisation
    p1 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(P*Psc/1e6)', xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:inferno, title="Pressure [MPa]")
    p2 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(τII*Psc/1e6)', xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), c=:inferno, title="Deviatoric stress inv [MPa]")
    p3 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array(T*Tsc.-273.15)', xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), xlabel="x [km]", ylabel="y [km]", c=:inferno, title="Temperature [°C]")
    p4 = scatter(Array(P*Psc/1e6),Array(τII.*Psc./1e6), zcolor = Array(abs.(ResP)), colorbar=true, legend=false, markerstrokewidth = 0.0)
    p4 = plot!(-[X0σ; X1σ; X2σ; -325e6/Psc].*Psc./1e6,[X0τ; X1τ; X2τ; 325e6/Psc*sind(φ)+Cm*cosd(φ)].*Psc./1e6, lw = 3)
    p5 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array((εII_pl./tsc))',  xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:roma, title="εII_pl")
    p6 = heatmap(x*Lsc/1e3, y*Lsc/1e3 , Array((ε_vol_pl./tsc))',  xlims=(x[1]*Lsc/1e3,x[end]*Lsc/1e3), ylims=(y[1]*Lsc/1e3,y[end]*Lsc/1e3), ylabel="y [km]", c=:roma, title="ε_vol_pl")
    plot(p1, p2, p3, p4, p5, p6 )
    plot!(size=(1920,1080), margin = 10Plots.mm); frame(anim)   # It is possible to display images in Real Time in Julia by "display(plot!(size=(1920,1080), margin = 10Plots.mm))", however it can significantly slow down the code
    gif(anim, PathToVisu*"MAGMA2Di_VEPCoolingPlasticIntrusion.gif", fps = 15)
    @printf("time = %1.3e yr \n", time*tsc/(365.25*24.0*3600.0))
    
    # Data Output
    if mod(it,nData) == 0 || it == 1
        SaveData = Dict(
            "x0"       => xO*Lsc,
            "y0"       => yO*Lsc,
            "rx"       => rx*Lsc,
            "ry"       => ry*Lsc,
            "px"       => px*Lsc,
            "py"       => py*Lsc,
            "tI"       => tI*tsc,
            "Lx"       => Lx*Lsc,
            "Ly"       => Ly*Lsc,
            "dV"       => dV*Vsc,
            "epsbg"    => εbg*εsc,
            "gval"     => gval*gsc,
            "alpha"    => α*αsc,
            "beta"     => β*βsc,
            "rhoRock"  => ρRock*ρsc,
            "rhoAir"   => ρAir*ρsc,
            "Tref"     => Tref*Tsc,     
            "Ttop"     => Ttop*Tsc,   
            "Tbottom"  => Tbottom*Tsc,  
            "Tmagma"   => Tmagma*Tsc, 
            "Cp"       => Cp*Cpsc,     
            "lambda"   => λ*λsc,
            "Qr"       => Qr*Qsc,  
            "Adis"     => Adis*Adissc,
            "ndis"     => ndis,
            "Edis"     => Edis*Esc,
            "Qr"       => Qr*Qsc,  
            "R"        => R*Rsc,
            "eta_min"  => η_min*ηsc,
            "eta_max"  => η_max*ηsc,
            "Edis"     => Edis*Esc,
            "F"        => F,
            "mu"       => μ*Psc,
            "phi"      => φ,
            "psi"      => ψ,
            "Cm"       => Cm*Psc,
            "C"        => Array(C)*Psc,
            "C_amp"    => C_amp*Psc,
            "S_T_to_C" => σ_T_to_C,
            "dS_T"     => δσ_T*Psc,
            "time"     => time*tsc,
            "nx"       => nx,
            "ny"       => ny,
            "dx"       => dx*Lsc,
            "dy"       => dy*Lsc,
            "rad"      => Array(rad),
            "P"        => Array(P*Psc),
            "T"        => Array(T*Tsc),
            "P_ini"    => Array(P_ini*Psc),
            "T_ini"    => Array(T_ini*Tsc),
            "rho_ini"  => Array(ρ_ini*ρsc),
            "rho"      => Array(ρ*ρsc),
            "rhoref"   => Array(ρref*ρsc),
            "eta"      => Array(η*ηsc),
            "eta_ve"   => Array(η_ve*ηsc),
            "eta_vep"  => Array(η_vep*ηsc),
            "Vx"       => Array(Vx*Vsc),
            "Vy"       => Array(Vy*Vsc),
            "Sxx"      => Array(σxx*Psc),
            "Syy"      => Array(σyy*Psc),
            "Szz"      => Array(σzz*Psc),
            "Sxy"      => Array(σxy*Psc),
            "tauII"    => Array(τII*Psc),
            "tauIIc"   => Array(τIIc*Psc),
            "Exx"      => Array(εxx*εsc),
            "Eyy"      => Array(εyy*εsc),
            "Ezz"      => Array(εzz*εsc),
            "Exy"      => Array(εxy*εsc),
            "EII"      => Array(εII*εsc),
            "E_vol"    => Array(∇V*εsc),
            "EII_vis"  => Array(εII_vis*εsc),
            "EII_pl"   => Array(εII_pl*εsc),
            "E_vol_pl" => Array(ε_vol_pl*εsc),
            "domain_pl"=> Array(domain_pl),
            "GII_pl"   => Array(γII_pl),
            "G_vol_pl" => Array(γ_vol_pl),
            "SH"       => Array(SH*Qsc),
            "SAir"     => Array(SAir),
            "SAirv"    => Array(SAirv),
            "C"        => Array(C*Psc),
            "x"        => Array(x*Lsc),       
            "y"        => Array(y*Lsc),         
            "xv"       => Array(xv*Lsc),        
            "yv"       => Array(yv*Lsc),        
            "xc"       => Array(xc*Lsc),        
            "yc"       => Array(yc*Lsc),        
            "xVx"      => Array(xVx*Lsc),       
            "yVx"      => Array(yVx*Lsc),       
            "xVy"      => Array(xVy*Lsc),       
            "yVy"      => Array(yVy*Lsc),       
            "xSxy"     => Array(xσxy*Lsc),     
            "ySxy"     => Array(yσxy*Lsc)) 
            DataFile = PathToData*"data_t"*string(it)*".mat"
        matwrite(DataFile,SaveData)
        end
    end    
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (4*2+6*1)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
                                                        # primary variables (read-write): P, ρ, Vx, Vy, T; history variables (read only): P_old, T_old, σxx_old, σyy_old, σzz_old, σxy_old 
                                                        # internal variables, such as viscosities, strain rates, or trial stresses are not included here
    wtime_it = wtime/(niter-10)                         # Execution time per iteration [s], incompressible initial guess and first 10 iterations in the time loop are excluded
    T_eff    = A_eff/wtime_it                           # Effective memory throughput [GB/s]
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, wtime, round(T_eff, sigdigits=2))
    return
end

## user defined macros
import ParallelStencil: INDICES
ix,  iy  = INDICES[1], INDICES[2]
# Anton's invariant definition (not cancelling the closest neighbours - averaging the squared values)
macro av2(A) esc(:( ($A[$ix  ,$iy  ]*$A[$ix  ,$iy  ] +
                     $A[$ix+1,$iy  ]*$A[$ix+1,$iy  ] +
                     $A[$ix  ,$iy+1]*$A[$ix  ,$iy+1] +
                     $A[$ix+1,$iy+1]*$A[$ix+1,$iy+1] )*0.25 )) end
macro max_x(A) esc(:( max($A[$ix,$iy],$A[$ix+1,$iy  ]) )) end
macro max_y(A) esc(:( max($A[$ix,$iy],$A[$ix  ,$iy+1]) )) end
macro max(A) esc(:( max($A[$ix,$iy],$A[$ix  ,$iy+1],$A[$ix+1  ,$iy],$A[$ix+1  ,$iy+1]) )) end
macro ηxMax(η,ηc) esc(:( max($η[$ix  ,$iy  ],$η[$ix+1,$iy  ],$ηc[$ix+1,$iy  ],$ηc[$ix+1,$iy+1] ) )) end
macro ηyMax(η,ηc) esc(:( max($η[$ix  ,$iy  ],$η[$ix  ,$iy+1],$ηc[$ix  ,$iy+1],$ηc[$ix+1,$iy+1] ) )) end
macro SerialMax(A) esc(:( max($A[i,j],$A[i,j+1],$A[i+1,j],$A[i+1,j+1]) )) end
macro SerialMin(A) esc(:( min($A[i,j],$A[i,j+1],$A[i+1,j],$A[i+1,j+1]) )) end

macro indE(ix,iy) esc(:( floor((xq[$ix,$iy]-Xmin)/dx) + 1.0  )) end
macro indW(ix,iy) esc(:( ceil( (xq[$ix,$iy]-Xmin)/dx) + 1.0  )) end
macro dX(ix,iy)   esc(:( (xq[$ix,$iy]-(Xmin+(@indE(ix,iy)-1.0)*dx))/dx )) end
macro indS(ix,iy) esc(:( floor((yq[$ix,$iy]-Ymin)/dy) + 1.0  )) end
macro indN(ix,iy) esc(:( ceil( (yq[$ix,$iy]-Ymin)/dy) + 1.0  )) end
macro dY(ix,iy)   esc(:( (yq[$ix,$iy]-(Ymin+(@indS(ix,iy)-1.0)*dy))/dy )) end

# parallel interp from 2D regular gridded data. Query points may have irregular coordinates.
@parallel_indices (ix,iy) function interp2!(Xmin::Data.Number,nx::Data.Number,dx::Data.Number,Ymin::Data.Number,ny::Data.Number,dy::Data.Number,Fint::Data.Array,F::Data.Array,xq::Data.Array,yq::Data.Array)
    if @indE(ix,iy) < 1.0 && @indS(ix,iy) < 1.0
        Fint[ix,iy]   = F[1      ,1      ] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1      ,1      ] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1      ,1      ] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[1      ,1      ] * (    @dX(ix,iy)) * (    @dY(ix,iy))                                                
    elseif @indE(ix,iy) < 1 && @indN(ix,iy) > ny
        Fint[ix,iy]   = F[1      ,Int(ny)] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1      ,Int(ny)] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1      ,Int(ny)] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[1      ,Int(ny)] * (    @dX(ix,iy)) * (    @dY(ix,iy))                        
    elseif @indW(ix,iy) > nx && @indS(ix,iy) < 1.0 
        Fint[ix,iy]  =  F[Int(nx),1      ] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx),1      ] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx),1      ] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(nx),1      ] * (    @dX(ix,iy)) * (    @dY(ix,iy))                            
    elseif @indW(ix,iy) > nx && @indN(ix,iy) > ny
        Fint[ix,iy]  =  F[Int(nx),Int(ny)] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx),Int(ny)] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx),Int(ny)] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(nx),Int(ny)] * (    @dX(ix,iy)) * (    @dY(ix,iy))                        
    elseif @indE(ix,iy) < 1.0
        Fint[ix,iy]   = F[1                ,Int(@indS(ix,iy))] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1                ,Int(@indS(ix,iy))] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[1                ,Int(@indN(ix,iy))] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[1                ,Int(@indN(ix,iy))] * (    @dX(ix,iy)) * (    @dY(ix,iy))                        
    elseif @indS(ix,iy) < 1
        Fint[ix,iy]  =  F[Int(@indE(ix,iy)),1                ] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),1                ] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indE(ix,iy)),1                ] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),1                ] * (    @dX(ix,iy)) * (    @dY(ix,iy))                            
    elseif @indW(ix,iy) > nx
        Fint[ix,iy]  =  F[Int(nx)          ,Int(@indS(ix,iy))] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx)          ,Int(@indS(ix,iy))] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(nx)          ,Int(@indN(ix,iy))] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(nx)          ,Int(@indN(ix,iy))] * (    @dX(ix,iy)) * (    @dY(ix,iy))
    elseif @indN(ix,iy) > ny
        Fint[ix,iy]  =  F[Int(@indE(ix,iy)),Int(ny)          ] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),Int(ny)          ] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indE(ix,iy)),Int(ny)          ] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),Int(ny)          ] * (    @dX(ix,iy)) * (    @dY(ix,iy))
    else
        Fint[ix,iy]  =  F[Int(@indE(ix,iy)),Int(@indS(ix,iy))] * (1.0-@dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),Int(@indS(ix,iy))] * (    @dX(ix,iy)) * (1.0-@dY(ix,iy)) +
                        F[Int(@indE(ix,iy)),Int(@indN(ix,iy))] * (1.0-@dX(ix,iy)) * (    @dY(ix,iy)) +
                        F[Int(@indW(ix,iy)),Int(@indN(ix,iy))] * (    @dX(ix,iy)) * (    @dY(ix,iy))
    end
    return
end

## ParallelStencil routines (GPU kernels)

@parallel function compute_timesteps_ic!(dτVx::Data.Array, dτVy::Data.Array, dτP::Data.Array, η::Data.Array, ηc::Data.Array, CLFV::Data.Number, CLFP::Data.Number, βn::Data.Number, min_dxy2::Data.Number, max_nxy::Data.Number)
    @all(dτVx) = min_dxy2/@ηxMax(η,ηc)*(1.0+βn)/4.1*CLFV
    @all(dτVy) = min_dxy2/@ηyMax(η,ηc)*(1.0+βn)/4.1*CLFV
    @all(dτP)  = 4.1*@all(η)*(1.0+βn)/max_nxy*CLFP
    return
end

@parallel function compute_timesteps_c!(dτVx::Data.Array, dτVy::Data.Array, dτP::Data.Array, dτT::Data.Array, η::Data.Array, ηc::Data.Array, ρRock::Data.Number, dt::Data.Number, λ::Data.Number, Cp::Data.Number, CLFV::Data.Number, CLFP::Data.Number, CLFT::Data.Number, min_dxy2::Data.Number, max_nxy::Data.Number)
    @all(dτVx) = min_dxy2/@ηxMax(η,ηc)/4.1*CLFV
    @all(dτVy) = min_dxy2/@ηyMax(η,ηc)/4.1*CLFV
    @all(dτP)  = 1.0/(2.0/dt + 1.0/(4.1*@all(η)/max_nxy*CLFP))
    @all(dτT)  = 1.0/(2.0/dt + 1.0/(min_dxy2/(λ/ρRock/Cp)/4.1*CLFT))
    return
end

@parallel function compute_P_ic!(∇V::Data.Array, P::Data.Array, dτP::Data.Array)
    @all(P)   = @all(P) - @all(∇V)*@all(dτP)
    return
end
   
@parallel function compute_ResP!(∇V::Data.Array, ResP::Data.Array, dPdτ::Data.Array, P::Data.Array, P_old::Data.Array, T::Data.Array, T_old::Data.Array, β::Data.Number, α::Data.Number, dampP::Data.Number, dt::Data.Number)
    @all(ResP) = -@all(∇V) - β*(@all(P)-@all(P_old))/dt + α*(@all(T)-@all(T_old))/dt
    @all(dPdτ) = dampP*@all(dPdτ) + @all(ResP)
    return
end

@parallel function Update_P!(dPdτ::Data.Array, P::Data.Array, dτP::Data.Array)
    @all(P)    =  @all(P) + @all(dPdτ)*@all(dτP)
    return
end

@parallel function compute_ρ!(ρ::Data.Array, P::Data.Array, T::Data.Array, ρref::Data.Array, Tref::Data.Number, α::Data.Number, β::Data.Number)
    @all(ρ)  = @all(ρref)*exp(β*@all(P)-α*(@all(T)-Tref))
    return
end

@parallel function update_ρ!(ρ::Data.Array, ρ_old::Data.Array, divV::Data.Array, dt::Data.Number)
    @all(ρ)  = @all(ρ_old)*exp(-@all(divV)*dt)
    return
end

@parallel function compute_εij!( ∇V::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, Vx::Data.Array, Vy::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(εxx)  = @d_xa(Vx)/dx - 1.0/3.0*@all(∇V)
    @all(εyy)  = @d_ya(Vy)/dy - 1.0/3.0*@all(∇V)
    @all(εzz)  =              - 1.0/3.0*@all(∇V)
    @inn(εxy)  = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    return
end

@parallel function compute_εij_ve!(εxx_ve::Data.Array, εyy_ve::Data.Array, εzz_ve::Data.Array, εxy_ve::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, σxx_old::Data.Array, σyy_old::Data.Array, σzz_old::Data.Array, σxy_old::Data.Array, P_old::Data.Array, μ::Data.Number, dt::Data.Number)
    @all(εxx_ve)  = @all(εxx) + (@all(σxx_old)+@all(P_old))/(2.0*μ*dt)
    @all(εyy_ve)  = @all(εyy) + (@all(σyy_old)+@all(P_old))/(2.0*μ*dt)
    @all(εzz_ve)  = @all(εzz) + (@all(σzz_old)+@all(P_old))/(2.0*μ*dt)
    @all(εxy_ve)  = @all(εxy) + (@all(σxy_old)            )/(2.0*μ*dt)
    return
end

@parallel function compute_εII_vis!(εII_vis::Data.Array, η::Data.Array, τII::Data.Array)
    @all(εII_vis ) = @all(τII )/(2.0*@all(η ))
    return
end

@parallel function compute_∇V!( ∇V::Data.Array, Vx::Data.Array, Vy::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)   = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    return
end

@parallel function compute_εII!(Mxx::Data.Array, Myy::Data.Array, Mzz::Data.Array, Mxy::Data.Array, MII::Data.Array, MIIc::Data.Array)
    @all(MII ) = sqrt(0.5*(  @all(Mxx)*@all(Mxx) + @all(Myy)*@all(Myy) + @all(Mzz)*@all(Mzz) ) + @av2(Mxy)           )
    @all(MIIc) = sqrt(0.5*(  @av2(Mxx)           + @av2(Myy)           + @av2(Mzz)           ) + @inn(Mxy)*@inn(Mxy) )
    return
end

@parallel function compute_τII!(Mxx::Data.Array, Myy::Data.Array, Mzz::Data.Array, Mxy::Data.Array, MII::Data.Array)
    @all(MII ) = sqrt(0.5*(  @all(Mxx)*@all(Mxx) + @all(Myy)*@all(Myy) + @all(Mzz)*@all(Mzz) ) + @av2(Mxy))
    return
end

@parallel function compute_τII_tr!(τII_tr::Data.Array, η_ve::Data.Array, εII_ve::Data.Array, τIIc_tr::Data.Array, ηc_ve::Data.Array, εIIc_ve::Data.Array)
    @all(τII_tr ) = 2.0*@all(η_ve )*@all(εII_ve)
    @all(τIIc_tr) = 2.0*@inn(ηc_ve)*@av(εII_ve )
    return
end

@parallel function compute_η!(η::Data.Array, ηc::Data.Array, T::Data.Array, εII::Data.Array, relη::Data.Number, Adis::Data.Number, ndis::Data.Number, Edis::Data.Number, R::Data.Number)
    @all(η)   = exp( (1.0-relη)*log(@all(η )) + relη*log( Adis^(-1.0/ndis) * @all(εII)^((1.0-ndis)/ndis) *exp(Edis/ndis/R/@all(T))))
    @inn(ηc)  = exp( (1.0-relη)*log(@inn(ηc)) + relη*log( Adis^(-1.0/ndis) * @av(εII )^((1.0-ndis)/ndis) *exp(Edis/ndis/R/@av(T) )))
    return
end

@parallel function const_η!(η::Data.Array, rad::Data.Array, η_const::Data.Number, η_inc::Data.Number)
    @all(η)   = η_const*(1.0-@all(rad)) + η_inc*(@all(rad))
    return
end

@parallel function const_ηc!(η::Data.Array, ηc::Data.Array)
    @inn(ηc)  = @av(η)
    return
end

@parallel function compute_η_ve!(η_ve::Data.Array, ηc_ve::Data.Array, η::Data.Array, ηc::Data.Array, dt::Data.Number, μ::Data.Number)
    @all(η_ve)   = 1.0/(1.0/@all(η ) + 1.0/(μ*dt))
    @inn(ηc_ve)  = 1.0/(1.0/@inn(ηc) + 1.0/(μ*dt))
    return
end

@parallel function StrainRateResidual!(ResτII::Data.Array, εII_ve::Data.Array, εII_pl::Data.Array, η_ve::Data.Array, τII::Data.Array, η::Data.Array, dt::Data.Number, μ::Data.Number)
    @all(ResτII) = (@all(εII_ve) - @all(τII)/(2.0*@all(η)) - @all(τII)/(2.0*μ*dt) - @all(εII_pl))/@all(εII_ve)
    return
end

@parallel function η_cutoff!(η::Data.Array, ηc::Data.Array, η_min::Data.Number, η_max::Data.Number, SAir::Data.Array, SAirv::Data.Array)
    @all(η)   = (@all(η )<=η_max)*@all(η ) + (@all(η )>η_max)*η_max
    @all(η)   = (@all(η )>=η_min)*@all(η ) + (@all(η )<η_min)*η_min
    @all(η)   = @all(η)*@all(SAir) + η_min*(1.0-@all(SAir))
    @inn(ηc)  = (@inn(ηc)<=η_max)*@inn(ηc) + (@inn(ηc)>η_max)*η_max
    @inn(ηc)  = (@inn(ηc)>=η_min)*@inn(ηc) + (@inn(ηc)<η_min)*η_min
    @inn(ηc)  = @inn(ηc)*@inn(SAirv) + η_min*(1.0-@inn(SAirv))
    return
end

@parallel function compute_σ!(P::Data.Array, σxx::Data.Array, σyy::Data.Array, σzz::Data.Array, σxy::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array,  η::Data.Array, ηc::Data.Array, ∇V::Data.Array, βn::Data.Number)
    @all(σxx ) = -@all(P) + 2.0*@all(η )*(@all(εxx ) + βn*@all(∇V))
    @all(σyy ) = -@all(P) + 2.0*@all(η )*(@all(εyy ) + βn*@all(∇V))
    @all(σzz ) = -@all(P) + 2.0*@all(η )*(@all(εzz ) + βn*@all(∇V))
    @inn(σxy ) =            2.0*@inn(ηc)*(@inn(εxy )              )
    return
end

@parallel_indices (ix, iy) function update_plastic_corrections!(F_tr::Data.Array, τII_tr::Data.Array, P_tr::Data.Array, εII_pl::Data.Array, ε_vol_pl::Data.Array,  εII_pl_old::Data.Array, ε_vol_pl_old::Data.Array, domain_pl::Data.Array, η_ve::Data.Array, η_vpl::Data.Number, dt::Data.Number,  β::Data.Number, φ::Data.Number, ψ::Data.Number, C::Data.Array, σ_T::Data.Array, δσ_T::Data.Number, x1σ::Data.Array, x1τ::Data.Array, x2σ::Data.Array, x2τ::Data.Array, relpl::Data.Number)
    F_tr[ix,iy]   = max(τII_tr[ix,iy] - P_tr[ix,iy]*sind(φ) - C[ix,iy]*cosd(φ) , τII_tr[ix,iy] - P_tr[ix,iy] - σ_T[ix,iy] , - P_tr[ix,iy] - (σ_T[ix,iy] - δσ_T) )
    if F_tr[ix,iy] > 0.0
        if τII_tr[ix,iy] <= x1τ[ix,iy]                                                                                                                                                                                  # tensile pressure cutoff
            εII_pl[ix,iy]    = relpl*0.0
            ε_vol_pl[ix,iy]  = relpl*(-P_tr[ix,iy]-(σ_T[ix,iy] - δσ_T))/(dt/β + 2.0/3.0*η_vpl)
            domain_pl[ix,iy] = 1.0
        elseif x1τ[ix,iy] < τII_tr[ix,iy] <= (η_ve[ix,iy] + η_vpl)/(dt/β + 2.0/3.0*η_vpl)*(-P_tr[ix,iy] - x1σ[ix,iy]) + x1τ[ix,iy]                                                                                      # 1st corner
            εII_pl[ix,iy]    = relpl*(τII_tr[ix,iy]-x1τ[ix,iy])/(η_ve[ix,iy] + η_vpl)/2.0
            ε_vol_pl[ix,iy]  = relpl*(-P_tr[ix,iy]-x1σ[ix,iy])/(dt/β + 2.0/3.0*η_vpl)
            domain_pl[ix,iy] = 2.0
        elseif (η_ve[ix,iy] + η_vpl)/(dt/β + 2.0/3.0*η_vpl)*(-P_tr[ix,iy] - x1σ[ix,iy]) + x1τ[ix,iy] < τII_tr[ix,iy] <= (η_ve[ix,iy] + η_vpl)/(dt/β + 2.0/3.0*η_vpl)*(-P_tr[ix,iy] - x2σ[ix,iy]) + x2τ[ix,iy]           # mode-1
            εII_pl[ix,iy]    = relpl*(τII_tr[ix,iy] - P_tr[ix,iy] - σ_T[ix,iy] )/(η_ve[ix,iy] + η_vpl*(1.0 + 2.0/3.0) + dt/β)/2.0*1.0                                                     # λ/2 * ∂Q/∂τ
            ε_vol_pl[ix,iy]  = relpl*(τII_tr[ix,iy] - P_tr[ix,iy] - σ_T[ix,iy] )/(η_ve[ix,iy] + η_vpl*(1.0 + 2.0/3.0) + dt/β)    *1.0                                                     # λ   * ∂Q/∂σ
            domain_pl[ix,iy] = 3.0 
        elseif (η_ve[ix,iy] + η_vpl)/(dt/β + 2.0/3.0*η_vpl)*(-P_tr[ix,iy] - x2σ[ix,iy]) + x2τ[ix,iy] < τII_tr[ix,iy] <= (η_ve[ix,iy] + η_vpl)/((dt/β + 2.0/3.0*η_vpl)*sind(ψ))*(-P_tr[ix,iy] - x2σ[ix,iy]) + x2τ[ix,iy] # 2nd corner
            εII_pl[ix,iy]    = relpl*(τII_tr[ix,iy]-x2τ[ix,iy])/(η_ve[ix,iy] + η_vpl)/2.0
            ε_vol_pl[ix,iy]  = relpl*(-P_tr[ix,iy]-x2σ[ix,iy])/(dt/β + 2.0/3.0*η_vpl)
            domain_pl[ix,iy] = 4.0
        elseif (η_ve[ix,iy] + η_vpl)/((dt/β + 2.0/3.0*η_vpl)*sind(ψ))*(-P_tr[ix,iy] - x2σ[ix,iy]) + x2τ[ix,iy] < τII_tr[ix,iy]                                                                                          # Drucker Prager
            εII_pl[ix,iy]    = relpl*(τII_tr[ix,iy] - P_tr[ix,iy]*sind(φ) - C[ix,iy]*cosd(φ))/(η_ve[ix,iy] + η_vpl*(1.0 + 2.0/3.0*sind(φ)*sind(ψ)) + dt/β*sind(ψ)*sind(φ))/2.0*1.0        # λ/2 * ∂Q/∂τ
            ε_vol_pl[ix,iy]  = relpl*(τII_tr[ix,iy] - P_tr[ix,iy]*sind(φ) - C[ix,iy]*cosd(φ))/(η_ve[ix,iy] + η_vpl*(1.0 + 2.0/3.0*sind(φ)*sind(ψ)) + dt/β*sind(ψ)*sind(φ))    *sind(ψ)    # λ   * ∂Q/∂σ
            domain_pl[ix,iy] = 5.0 
        end
        # can help convergence if viscoplastic regularization is not sufficient (e.g., relpl is too large), or trial stresses and trial pressure are far away from the yield (e.g., stress increment is too large because of too large time steps)
        #εII_pl[ix,iy]   = 0.003*εII_pl[ix,iy] + (1.0-0.003)*εII_pl_old[ix,iy]       
        #ε_vol_pl[ix,iy] = 0.003*ε_vol_pl[ix,iy]   + (1.0-0.003)*ε_vol_pl_old[ix,iy]
    else
        εII_pl[ix,iy]    = 0.0
        ε_vol_pl[ix,iy]  = 0.0
        domain_pl[ix,iy] = 0.0
    end
    return 
end

@parallel function plastic_stress_corrections!( σxx::Data.Array, σyy::Data.Array, σzz::Data.Array, σxy::Data.Array, σxx_tr::Data.Array, σyy_tr::Data.Array, σzz_tr::Data.Array, σxy_tr::Data.Array, τII::Data.Array, τIIc::Data.Array, τII_tr::Data.Array, τIIc_tr::Data.Array, P_tr::Data.Array, P::Data.Array, εII_pl::Data.Array, ε_vol_pl::Data.Array, η_ve::Data.Array, ηc_ve::Data.Array, dt::Data.Number, β::Data.Number) 
    @all(P)     = @all(P_tr) + dt/β*@all(ε_vol_pl)
    @all(τII)   = @all(τII_tr ) - 2.0*@all(η_ve )*@all(εII_pl)
    @all(τIIc)  = @all(τIIc_tr) - 2.0*@inn(ηc_ve)*@av(εII_pl )
    @all(σxx)   = -@all(P) + (@all(σxx_tr)+@all(P_tr))*(1.0 - 2.0*@all(η_ve )*@all(εII_pl)/@all(τII_tr))
    @all(σyy)   = -@all(P) + (@all(σyy_tr)+@all(P_tr))*(1.0 - 2.0*@all(η_ve )*@all(εII_pl)/@all(τII_tr))
    @all(σzz)   = -@all(P) + (@all(σzz_tr)+@all(P_tr))*(1.0 - 2.0*@all(η_ve )*@all(εII_pl)/@all(τII_tr))
    @inn(σxy)   =            (@inn(σxy_tr)           )*(1.0 - 2.0*@inn(η_ve )*@av(εII_pl )/@av(τII_tr ))
    return
end

@parallel function CalculateVEP_viscosity!(τII::Data.Array, εII::Data.Array, η_vep::Data.Array, τIIc::Data.Array, ηc_vep::Data.Array)
    @all(η_vep)  = @all(τII)/2.0/@all(εII)
    @inn(ηc_vep) = @all(τIIc)/2.0/@av(εII)
    return
end

@parallel function compute_dV!(RVx::Data.Array, RVy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, ρ::Data.Array, σxx::Data.Array, σyy::Data.Array, σxy::Data.Array, dampX::Data.Number, dampY::Data.Number, gval::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(RVx)    = (@d_xa(σxx)/dx + @d_yi(σxy)/dy) 
    @all(RVy)    = (@d_ya(σyy)/dy + @d_xi(σxy)/dx + gval*@av_ya(ρ) )
    @all(dVxdτ)  = dampX*@all(dVxdτ) + @all(RVx)
    @all(dVydτ)  = dampY*@all(dVydτ) + @all(RVy)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn_x(Vx) = @inn_x(Vx) + @all(dVxdτ)*@all(dτVx)
    @inn_y(Vy) = @inn_y(Vy) + @all(dVydτ)*@all(dτVy)
    return
end

@parallel function compute_OldPos!(xc_old::Data.Array, yc_old::Data.Array, xc::Data.Array, yc::Data.Array, Vx::Data.Array, Vy::Data.Array, dt::Data.Number)
    @all(xc_old) = @all(xc) - @av_xa(Vx)*dt
    @all(yc_old) = @all(yc) - @av_ya(Vy)*dt
    return
end

@parallel function compute_SH!(SH::Data.Array, τII::Data.Array, εII_vis::Data.Array, εII_pl::Data.Array, P::Data.Array, ε_vol_pl::Data.Array)
    @all(SH) = 2.0*@all(τII)*(@all(εII_vis) + @all(εII_pl)) - @all(P)*@all(ε_vol_pl)
    return
end

@parallel function compute_heatflux!(T::Data.Array, qx::Data.Array, qy::Data.Array, λ::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn_x(qx) = -λ*@d_xa(T)/dx
    @inn_y(qy) = -λ*@d_ya(T)/dy
    return  
end

@parallel function InitialT!(dTdt::Data.Array,T::Data.Array, qx::Data.Array, qy::Data.Array, ρ::Data.Array, Cp::Data.Number, Qr::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number, SAir::Data.Array)  
    @all(dTdt) = (-@d_xa(qx)/dx - @d_ya(qy)/dy + Qr)/(@all(ρ)*Cp)*@all(SAir)
    @all(T) = @all(T) + @all(dTdt)*dt
    return
end

@parallel function TAnomaly!(T::Data.Array, rad::Data.Array, Tmagma::Data.Number)  
    @all(T) = @all(T) *(1.0-@all(rad)) + Tmagma*@all(rad)
    return
end

@parallel function compute_T!(dTdt::Data.Array,ResT::Data.Array,T::Data.Array, T_old::Data.Array, P::Data.Array, P_old::Data.Array, qx::Data.Array, qy::Data.Array, ρ::Data.Array, SH::Data.Array, dTdτ::Data.Array, dτT::Data.Array, α::Data.Number, Cp::Data.Number, Qr::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number, dampT_c::Data.Number, SAir::Data.Array)  
    @all(dTdt) = (-@d_xa(qx)/dx - @d_ya(qy)/dy + α*@all(T)*(@all(P)-@all(P_old))/dt + @all(SH) + Qr)/(@all(ρ)*Cp)*@all(SAir)
    @all(ResT) = -(@all(T) -@all(T_old))/dt + @all(dTdt);
    @all(dTdτ) = dampT_c*@all(dTdτ) + @all(ResT)
    @all(T)    = @all(T) + @all(dTdτ)*@all(dτT)
    return
end

@parallel function LinearTermsInT!(LinT::Data.Array,T::Data.Array, T_old::Data.Array, qx::Data.Array, qy::Data.Array, ρ::Data.Array, Cp::Data.Number, Qr::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number)  
    @all(LinT) = (-@d_xa(qx)/dx - @d_ya(qy)/dy + Qr)/(@all(ρ)*Cp)
    @all(LinT) = -(@all(T) -@all(T_old))/dt + @all(LinT)
    return
end

@parallel function TimeIntegration(A::Data.Array, B::Data.Array , dt::Data.Number)
    @all(A) = @all(A) + @all(B)*dt
    return
end

@parallel_indices (ix) function Surface_T!(T::Data.Array, Ttop::Data.Number)
    T[ix, end] = Ttop
    return
end

@parallel_indices (ix) function Bottom_T!(qy::Data.Array, T::Data.Array, Tbottom::Data.Number, λ::Data.Number, dy::Data.Number)
    qy[ix, 1] = -2.0*λ*(T[ix,1]-Tbottom)/dy
    return
end

@parallel function smooth!(A::Data.Array)
        @inn(A) = @inn(A) + 1.0/8.1*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function CopyParallelArray!(Original::Data.Array, Copy::Data.Array)
    @all(Copy) = @all(Original)
    return
end

@parallel_indices (ix) function FreeSurface_σyy!(σyy::Data.Array)
    σyy[ix, end] = 0.0
    return
end
@parallel_indices (ix) function FreeSurface_σxy!(σxy::Data.Array)
    σxy[ix, end] =  -σxy[ix, end-1]
    return
end

@parallel_indices (ix) function FreeSurface_Vy!(Vy::Data.Array,P::Data.Array,η::Data.Array,Vx::Data.Array,dx::Data.Number,dy::Data.Number,)
    Vy[ix, end] = Vy[ix, end-1] + 3.0/2.0*(P[ix, end]/(2.0*η[ix, end]) + 1.0/3.0 * (Vx[ix+1, end]-Vx[ix, end])/dx)*dy
    return
end
@parallel_indices (ix) function FreeSurface_Vy_ve!(Vy::Data.Array,P::Data.Array,σyy_old::Data.Array,P_old::Data.Array,η::Data.Array,Vx::Data.Array,μ::Data.Number,dt::Data.Number,dx::Data.Number,dy::Data.Number,)
    Vy[ix, end] = Vy[ix, end-1] + 3.0/2.0*(P[ix, end]/(2.0*η[ix, end]) - (σyy_old[ix, end]+P_old[ix, end])/(2.0*μ*dt) + 1.0/3.0 * (Vx[ix+1, end]-Vx[ix, end])/dx)*dy
    return
end
@parallel_indices (ix) function FreeSurface_Vy_vep!(Vy::Data.Array,P::Data.Array,σyy_old::Data.Array,P_old::Data.Array,η::Data.Array,Vx::Data.Array,εII_pl::Data.Array,μ::Data.Number,dt::Data.Number,dx::Data.Number,dy::Data.Number)
    Vy[ix, end] = Vy[ix, end-1] + 3.0/2.0*(P[ix, end]/(2.0*η[ix, end]*(1.0-εII_pl[ix, end])) - (σyy_old[ix, end]+P_old[ix, end])/(2.0*μ*dt) + 1.0/3.0 * (Vx[ix+1, end]-Vx[ix, end])/dx)*dy
    return
end
@parallel function set_far_field!(Vx::Data.Array, εbg::Data.Number, xVx::Data.Array)
    @all(Vx) = εbg*@all(xVx)
    return
end
@parallel function Cohesion_softening!(C::Data.Array, C_old::Data.Array, E_vol_pl::Data.Array, Soft::Data.Number, dt::Data.Number, C_min::Data.Number)
    @all(C)        = @all(C_old) - Soft*@all(E_vol_pl)*dt
    @all(C)        = @all(C)*(@all(C)>C_min) + C_min*(@all(C)<C_min)
    return
end

@parallel function set_plastic_parameters!(C::Data.Array, σ_T::Data.Array, x1σ::Data.Array, x1τ::Data.Array,x2σ::Data.Array, x2τ::Data.Array, δσ_T::Data.Number, φ::Data.Number, ψ::Data.Number, σ_T_to_C::Data.Number)
    @all(σ_T)      =  σ_T_to_C*@all(C)
    @all(x1σ)      =  @all(σ_T) - δσ_T                              # σm at the intersection of cutoff and Mode-1
    @all(x1τ)      = -@all(x1σ) +  @all(σ_T)                        # τII at the intersection of cutoff and Mode-1
    @all(x2σ)      = (@all(σ_T)-@all(C)*cosd(φ))/(1.0-sind(φ))      # σm at the intersection of Drucker-Prager and Mode-1
    @all(x2τ)      = -@all(x2σ) + @all(σ_T)                         # τII at the intersection of Drucker-Prager and Mode-1
    return
end
@parallel function SAirv!(SAirv::Data.Array, SAir::Data.Array)
    @inn(SAirv) = @all(SAir)
    return
end

function IntrusionCoordinates(Lx,dx,nx,Ly,dy,ny,Ymax,SubGridRes,xO,yO,rx,ry,px,py,event)
    xrad         = [-Lx/2.0+dx/SubGridRes/2.0    + (ix-1)*dx/SubGridRes for ix = 1:nx*SubGridRes   ]
    yrad         = [-(Ly-Ymax)+dy/SubGridRes/2.0 + (iy-1)*dy/SubGridRes for iy = 1:ny*SubGridRes   ]
    xcrad        = [xrad[ix] for ix=1:nx*SubGridRes , iy=1:ny*SubGridRes ]
    ycrad        = [yrad[iy] for ix=1:nx*SubGridRes , iy=1:ny*SubGridRes ]
    extrad       = zeros(nx*SubGridRes,ny*SubGridRes)
    extrad      .= ((xcrad.-xO[event])./rx[event]).^px[event] .+ ((ycrad.-yO[event])./ry[event]).^py[event]
    extrad      .= extrad.<=1.0
    rad          = [mean(extrad[SubGridRes*(ix-1)+1:SubGridRes*ix,SubGridRes*(iy-1)+1:SubGridRes*iy]) for ix=1:nx, iy=1:ny]
    extrad       = nothing
    GC.gc()
    return rad
end

@time MAGMA_2D_PT_VEPCoolingPlasticIntrusion()