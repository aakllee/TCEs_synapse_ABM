# A reduction of the Liu et al. 2023 ABM to an ODEs model.

using DifferentialEquations
using Distributions
using Match
using Plots
using PhysicalConstants
using ThreadsX
using Memoization

ngmL_to_M(conc, MW) = conc * 1000 * 1e-9 / MW

"Pre-allocated cache for ODE solving." 
struct ODECache{T}
    prob::ODEProblem
    xstart::Vector{T}
    bondtime_dist::Uniform{T}
end
function create_ode_cache(rate_constants, T)
    xstart = zeros(5)
    f = ODEFunction(ode_binding!; jac=jac_binding!)
    prob = ODEProblem(f, zeros(5), (0.0, 60.0), rate_constants) # 60 seconds by default
    return ODECache{T}(prob, xstart, Uniform(0.1, 5.0))
end

@kwdef mutable struct Parameters{T<:Real}
    n_effector_0::T  = 1e6 # Initial effector cells, 1/mL.
    n_target_0::T    = 1e6 # Initial target cells, 1/mL.
    
    # Simulation parameters
    Na::T = PhysicalConstants.CODATA2022.N_A.val # Avogradro constant, 1/mol
    dt::Int64 = 1 # Timestep, mins.

    # Model parameters
    TCE_conc::T    = 1.0     # [TCE], flag_nM ? nM : ng/mL.
    flag_nM::Bool  = false
    MW::T          = 55_000  # TCE molecular weight, g/mol. 
    TCR_dist::T    = 66_299  # TCR antigen mean, molecules/cell.
    TAA_dist::T    = 144_866 # TAA antigen mean, molecules/cell.
    KD_TCR::T      = 2.6e-7  # TCR binding affinity, M. 
    KD_TAA::T      = 1.49e-9 # TAA binding affinity, M.
    S_effector::T  = 4*pi * (5.0^2) * 1.8 # Effector surface area, um^2.
    S_target::T    = 4*pi * (6.0^2) * 1.8 # Target surface area, um^2.
    D::T           = 0.83  # Cell diffusion coefficient, um^2/s.
    R_system::T    = 6200  # Spherical diameter of reaction system, um.
    R_target::T    = 6.0   # Radius of target cell, um.
    R_effector::T  = 5.0   # Radius of effector cell, um.
    kint::T        = 0.002 # Internalisation rate of TAA, unitless.
    beta::T        = 0.028 # 0.033 # Binding constant, unitless.
    Sc1::T         = 5.0   # Synapse surface contact area, um^2.
    
    # Spatial constants
    fETE::T        = 0.75
    fTET::T        = 0.66
    fETET::T       = 0.75
    fETEE::T       = 0.5
    fTETT::T       = 0.33
    fTETE::T       = 0.67

    # Variables
    equilibrium_TCR_free::T    = 0.0 # Free TCR at equilibrium, 1/um^2.
    equilibrium_TCR_binary::T  = 0.0 # TCR binary complex at equilibrium, 1/um^2.
    equilibrium_TAA_free::T    = 0.0 # Free TAA at equilibrium, 1/um^2.
    equilibrium_TAA_binary::T  = 0.0 # TAA binary complex at equilibrium, 1/um^2.
    konA::T  # = NaN # To be calculated by calculate_rate_constants
    konB::T  # = NaN # To be calculated by calculate_rate_constants
    koffA::T # = NaN # To be calculated by calculate_rate_constants
    koffB::T # = NaN # To be calculated by calculate_rate_constants
    encounter_probability_ET::T   = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_TET::T  = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_ETET::T = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_TETT::T = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_ETE::T  = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_ETEE::T = 0.0 # Set by calculate_encounter_probabilities!(model)
    encounter_probability_TETE::T = 0.0 # Set by calculate_encounter_probabilities!(model)
    cache::ODECache{T} = create_ode_cache((; konA, konB, koffA, koffB), T)
end

function run_model(time;
                   n_effector_0 = 1e6, # Initial effector cells, 1/mL.
                   n_target_0   = 1e6, # Initial target cells, 1/mL.
                   TCE_conc     = 1.0,
                   flag_nM      = false,
                   MW           = 55_000,  # TCE molecular weight, g/mol. 
                   TCR_dist     = 66_299,  # TCR antigen mean, molecules/cell.
                   TAA_dist     = 144_866, # TAA antigen mean, molecules/cell.
                   KD_TCR       = 2.6e-7,  # TCR binding affinity, M. 
                   KD_TAA       = 1.49e-9, # TAA binding affinity, M.
                   )
    rate_constants = calculate_rate_constants(KD_TCR, KD_TAA; Na = 6.022e23)
    p = Parameters{Float64}(;
                            n_effector_0, n_target_0,
                            TCE_conc, flag_nM,
                            MW, TCR_dist, TAA_dist, KD_TCR, KD_TAA,
                            rate_constants...)
    calculate_equilibrium!(n_effector_0, n_target_0, p)

    u0 = zeros(Float64, max(idx_n_conjugates()..., idx_n_target_free()..., idx_n_effector_free()...))
    u0[idx_n_effector_free()] = n_effector_0
    u0[idx_n_target_free()]   = n_target_0
    tspan = (0, time)
    prob = ODEProblem(reduced_liu_model!, u0, tspan, p)
    sol = solve(prob, BS3(); abstol=1e-3, reltol=1e-2)
    return sol
end

function reproduce_liu_fig3()
    liufig3b_data = [0.6487015067924768  1.2831858407079646; 4.887775298393741   2.256637168141593; 19.434038570303596  4.601769911504424; 49.42251490751028   11.283185840707967; 98.54287131604329   12.52212389380531; 196.6991032270088   12.52212389380531; 392.8574323347438   11.858407079646017; 997.1542887677606   0.7079646017699125; 1990.1620644143597  0.8407079646017688]

    TCE_concs = liufig3b_data[:,1]
    ctoxs = liufig3b_data[:,2]

    TCE_concs_sim = 10 .^ range(log10(minimum(TCE_concs))-1, log10(maximum(TCE_concs))+1, 50)

    # Solve model for each concentration
    sols = ThreadsX.map(TCE_conc -> run_model(60; TCE_conc), TCE_concs_sim)

    synapse_effector = [
        begin
            # Count bound effectors
            n_conjugates_vec = sol[end][idx_n_conjugates()]
            n_conjugates = reshape(n_conjugates_vec, 3, 3)'
            n_free = sol[end][idx_n_effector_free()]
            n_bound = 0
            for nE in 1:3, nT in 1:3
                n_bound += n_conjugates[nE, nT] * nE
            end
            n_total = n_bound + n_free
            n_bound / n_total * 100
        end for sol in sols]

    fig = plot(; xlabel = "[Blinatumomab], ng/mL", ylabel = "Effector % engaged", xscale = :log10)
    scatter!(fig, TCE_concs, ctoxs, c = 1)
    plot!(fig, TCE_concs_sim, synapse_effector, c = 1)
    return fig
end


####################################################################################

idx_n_effector_free() = 10
idx_n_target_free()   = 11
idx_n_conjugates()    = 1:9

@fastmath TCR_hill(TCR_binary, t; gamma = 0.9, h = 0.7) = 1 / (1 + 0.1 * TCR_binary^gamma * min(t, 60)^h)

function reduced_liu_model!(du, u, p, t)
    n_conjugates = reshape(u[idx_n_conjugates()], 3, 3)'
    n_effector_free = u[idx_n_effector_free()]
    n_target_free   = u[idx_n_target_free()]

    d_effector_free = 0
    d_target_free   = 0
    d_conjugates    = zeros(3, 3)

    # Apply CD3 downregulation
    ratio = TCR_hill(p.equilibrium_TCR_binary, t)
    TCR_free   = p.equilibrium_TCR_free   * ratio
    TCR_binary = p.equilibrium_TCR_binary * ratio

    # Apply CD19 internalisation
    TAA_free   = p.equilibrium_TAA_free
    TAA_binary = p.equilibrium_TAA_binary * exp(-p.kint * t)
    
    # Calculate encounter rates.
    r_encounters = calculate_encounter_rates(n_effector_free, n_target_free, p)

    # Calculate binding rates. 
    r_bindings = calculate_binding_probability!(TCR_free, TCR_binary, TAA_free, TAA_binary, p)

    # Formation of new single conjugates.
    d_conjugates[1,1] = r_encounters.single * r_bindings * n_effector_free
    d_effector_free -= d_conjugates[1,1]
    d_target_free   -= d_conjugates[1,1]

    # Formation of multiple conjugates.
    @inbounds for n_effector in 1:2, n_target in 1:2
        # Get numbers of this conjugate.
        n_conjugate = n_conjugates[n_effector, n_target]

        # Get rate of binding of effectors and targets.
        nr_effector = n_conjugate * r_bindings  * r_encounters.effector[n_effector, n_target]
        nr_target   = n_conjugate * r_bindings  * r_encounters.target[n_effector, n_target]

        # Remove effectors/targets from free pool
        d_effector_free -= nr_effector
        d_target_free   -= nr_target
        
        # Transfer from this population to higher order conjugate populations.
        d_conjugates[n_effector+1, n_target]   += nr_effector
        d_conjugates[n_effector,   n_target+1] += nr_target
        d_conjugates[n_effector,   n_target]   -= nr_effector
        d_conjugates[n_effector,   n_target]   -= nr_target
    end

    du[idx_n_effector_free()] = d_effector_free
    du[idx_n_target_free()]   = d_target_free
    du[idx_n_conjugates()] .= reshape(d_conjugates', 9)
    return nothing
end

"Solves free antigen and binary complex concentration levels at eqm.
Returns values in molecules per um^2 (on cell surface)." 
function calculate_equilibrium!(n_effector, n_target, p)
    (; TCE_conc, flag_nM, MW, Na, TCR_dist, TAA_dist, KD_TCR, KD_TAA, S_effector, S_target) = p

    # Get average antigen numbers
    TCR_mean = TCR_dist
    TAA_mean = TAA_dist
    KDA = KD_TCR
    KDB = KD_TAA
    
    # Convert all to M
    Ytot = flag_nM ? TCE_conc * 1e-9 : ngmL_to_M(TCE_conc, MW) #  Y_conc * 1000 * 1e-9 / MW
    Atot = n_effector * TCR_mean * 1000 / Na
    Btot = n_target   * TAA_mean * 1000 / Na
    
    aA = Atot / Ytot
    aB = Btot / Ytot
    ztest = aA + aB
    kA = KDA / Ytot
    kB = KDB / Ytot
    
    b = -(2 + kA + kB + aA + aB)
    c = 1 + 2aA + 2aB + kA + kB + kB*aA + kA*aB + kA*kB
    d = -(kB*aA + kA*aB + aA + aB)
    Q = (3c - b^2) / 9
    Rr_c = (9b*c - 27d - 2*(b^3)) / 54
    Th = acos(Rr_c / sqrt(-Q^3))
    
    z1 = -b/3 + 2*sqrt(-Q) * cos(Th/3)
    z2 = -b/3 + 2*sqrt(-Q) * cos((Th + 2*pi)/3)
    z3 = -b/3 + 2*sqrt(-Q) * cos((Th + 4*pi)/3)
    
    z = NaN
    @inbounds for zz in (z1, z2, z3)
        if zz < ztest && 0 < zz < 1
            z = zz
            break
        end
    end
    isnan(z) && throw("Error, z NaN in calculating equilibrium")

    # A + Y --(KDA)--> AY  |  B + Y --(KDB)--> YB
    A_M = Atot * KDA / (KDA + Ytot*(1 - z)) # Free TCR, M
    B_M = Btot * KDB / (KDB + Ytot*(1 - z)) # Free TAA, M
    AY_M = (Ytot * A_M / KDA) / (1 + (A_M / KDA) + (B_M / KDB)) # TCR binary complex, M
    YB_M = (Ytot * B_M / KDB) / (1 + (A_M / KDA) + (B_M / KDB)) # TAA binary complex, M

    # M --> molecules/cell
    A_cell  = (A_M * Na)  / (1000 * n_effector)
    B_cell  = (B_M * Na)  / (1000 * n_target)
    AY_cell = (AY_M * Na) / (1000 * n_effector)
    YB_cell = (YB_M * Na) / (1000 * n_target)

    # molecules/cell --> molecules/um^2
    p.equilibrium_TCR_free  = A_cell/S_effector
    p.equilibrium_TAA_free  = B_cell/S_target
    p.equilibrium_TCR_binary = AY_cell/S_effector
    p.equilibrium_TAA_binary = YB_cell/S_target

    return nothing
end

# TODO: add comments from original R model. Unchanged from original.
function calculate_rate_constants(KDA, KDB; Na = 6.022e23)
    kons   = 0.0001
    koffAs = kons * KDA * Na / 1e15
    koffBs = kons * KDB * Na / 1e15    
    DsB    = 50.0
    DsA    = 50.0
    DsY    = 50.0
    DmB    = 0.006
    DmA    = 0.01 - DmB
    Rab    = 0.005    
    dsf    = 4*pi * (DsB + DsY) * Rab
    dsr    = 3 * (DsB + DsY) / (Rab^2)
    dmf    = 2*pi * (DmB + DmA)
    dmr    = 2 * (DmB + DmA) / (Rab^2)
    esr    = (3 * dsr) / (4*pi^2)
    esf    = 0.04 * esr
    emr    = (3 * dmr) / (4*pi^2)
    emf    = 0.04 * emr    
    rsfA   = (kons * dsr * esr) / (dsf * esf - kons * (dsr + esf))
    rsfB   = (kons * dsr * esr) / (dsf * esf - kons * (dsr + esf))
    rsrA   = koffAs * (dsr * esr + rsfA * (dsr + esf)) / (dsr * esr)
    rsrB   = koffBs * (dsr * esr + rsfB * (dsr + esf)) / (dsr * esr)
    konA   = dmf * rsfA * emf / (dmr * emr + rsfA * (dmr + emf))
    koffA  = dmr * rsrA * emr / (dmr * emr + rsfA * (dmr + emf))
    konB   = dmf * rsfB * emf / (dmr * emr + rsfB * (dmr + emf))
    koffB  = dmr * rsrB * emr / (dmr * emr + rsfB * (dmr + emf))    
    return (; konA, konB, koffA, koffB)
end


################# ODEs #########################
@fastmath function ode_binding!(du, u, p, t)
    AYB, A, AY, B, YB = @view u[:]
    (; konA, konB, koffA, koffB) = p
    
    du[1] = konA*A*YB + konB*B*AY - koffA*AYB - koffB*AYB
    du[2] = koffA*AYB - konA*A*YB
    du[3] = koffB*AYB - konB*B*AY
    du[4] = koffB*AYB - konB*B*AY
    du[5] = koffA*AYB - konA*A*YB
    return nothing
end
# Jacobian of binding ODEs
@fastmath function jac_binding!(J, u, p, t)
    AYB, A, AY, B, YB = @view u[:]
    (; konA, konB, koffA, koffB) = p
    J[:,:] = [
        -(koffA+koffB)  konA*YB   konB*B   konB*AY   konA*A;
        koffA           -konA*YB  0        0         -konA*A;
        koffB           0         -konB*B  -konB*AY  0;
        koffB           0         -konB*B  -konB*AY  0;
        koffA           -konA*YB  0        0         -konA*A;
    ]
    return nothing
end


"Solves ODEs to calculate adhesion (binding) probability."
function calculate_binding_probability!(TCR_free, TCR_binary, TAA_free, TAA_binary, p)
    (; Sc1, beta) = p
    p.cache.xstart[:] .= 0.0, TCR_free, TCR_binary, TAA_free, TAA_binary

    # Instead of running for the full 60 seconds, we instead only run for the bond time.
    # The 60 second value is never actually used in the original Liu model. 
    bondtime = mean(p.cache.bondtime_dist)
    
    # Remake problem with new initial conditions and timespan.
    # Apparently there's an analytical solution anyway..?
    prob = remake(p.cache.prob; u0=p.cache.xstart, tspan = (0.0, bondtime))
    solver = BS3() # TRBDF2()
    sol = solve(prob, solver; abstol=1e-3, reltol=1e-2, dt=0.1,
                saveat=bondtime, save_everystep=false, dense=false)

    trimer_per_surface_area = sol[end][1] # AYB bonc conc at bondtime
    trimer = trimer_per_surface_area * Sc1 # Total AYB bond at bondtime

    return 1 - exp(-beta * trimer) # Pb
end

"Calculates encounter rates for each conjugate."
function calculate_encounter_rates(n_effector_free, n_target_free, p)
    (; D, R_system, R_effector, R_target, fTET, fETET, fTETT, fETE, fETEE, fTETE) = p

    dis = R_effector + R_target
    alpha = 1.0 / ((R_system^3) / (3 * D * dis) - 0.6 * (R_system^2 / D))
    
    an_E = alpha*60 * n_effector_free
    an_T = alpha*60 * n_target_free

    r_ET   = an_T               # E   + T
    r_TET  = an_T * fTET  * 2   # ET  + T
    r_ETET = an_T * fETET * 3   # ETT + E
    r_TETT = an_T * fTETT * 3   # ETT + T
    r_ETE  = an_E * fETE  * 2   # ET  + E
    r_ETEE = an_E * fETEE * 3   # ETE + E
    r_TETE = an_E * fTETE * 3   # ETE + T
    return (
        single   = r_ET,
        target   = [r_TET  r_TETT ; r_TETE  0.0],
        effector = [r_ETE  r_ETET ; r_ETEE  0.0],
    )
end

