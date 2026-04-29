##### Liu functions #####

"Solves free antigen and binary complex concentration levels at eqm.
Returns values in molecules per um^2 (on cell surface)." 
function calculate_equilibrium!(model)
    (; TCE_conc, flag_nM, MW, Na, TCR_dist, TAA_dist, KD_TCR, KD_TAA, S_effector, S_target) = abmproperties(model)

    n_effector = count_free_effectors(model)
    n_target   = count_free_targets(model)
    
    # Get average antigen numbers
    TCR_mean = float(mean(TCR_dist))
    TAA_mean = float(mean(TAA_dist))
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
    model.equilibrium_TCR_free  = A_cell/S_effector
    model.equilibrium_TAA_free  = B_cell/S_target
    model.equilibrium_TCR_binary = AY_cell/S_effector
    model.equilibrium_TAA_binary = YB_cell/S_target

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

function calculate_binding_probability!(effector, target, model)
    (; Sc1, beta) = abmproperties(model)
    if !(model.cache isa ODECache)
        model.cache = create_ode_cache(model.rate_constants, typeof(beta))
    end
    return calculate_binding_probability!(
        model.cache, effector.TCR_free, effector.TCR_binary,
        target.TAA_free, target.TAA_binary, Sc1, beta, model)
end

# It may be worth attempting a solve across all agents simultaneously,
# possibly on GPU via Ensemble solve.
# Easier may just be to find the analytical quartic solution.
"Solves ODEs to calculate adhesion (binding) probability."
function calculate_binding_probability!(cache, A, AY, B, YB, Sc1, beta, model)
    cache.xstart[:] .= 0.0, A, AY, B, YB

    # Instead of running for the full 60 seconds, we instead only run for the bond time.
    # The 60 second value is never actually used in the original Liu model. 
    bondtime = round(rand(abmrng(model), cache.bondtime_dist); digits=1)
    
    # Remake problem with new initial conditions and timespan.
    # Apparently there's an analytical solution anyway..?
    prob = remake(cache.prob; u0=cache.xstart, tspan = (0.0, bondtime))
    solver = BS3() # TRBDF2()
    sol = solve(prob, solver; abstol=1e-3, reltol=1e-2, dt=0.1,
                saveat=bondtime, save_everystep=false, dense=false)

    AYB = sol[end][1] # AYB bonc conc at bondtime
    NAYB = AYB * Sc1 # Total AYB bond at bondtime

    return 1 - exp(-beta * NAYB) # Pb
end

"Calculates encounter probabilities for each conjugate.
Here, E = effector, T = target."
function calculate_encounter_probabilities!(model)
    (; D, R_system, R_target, R_effector, fETE, fTET, fETEE, fETET, fTETT, fTETE) = abmproperties(model)
    # n_effector_single = count_cells(Effector, model)
    # n_target_single   = count_cells(Target,   model)

    n_effector_single = count_free_effectors(model)
    n_target_single   = count_free_targets(model)
    
    dis = R_target + R_effector
    alpha = 1.0 / ((R_system^3) / (3 * D * dis) - 0.6 * (R_system^2 / D))
    
    exp_target   = exp(-alpha * n_target_single   * 60)
    exp_effector = exp(-alpha * n_effector_single * 60)
    
    model.encounter_probability_ET   = (1 - exp_target)                 # E   + T
    model.encounter_probability_TET  = (1 - exp_target)   * fTET  * 2   # ET  + T
    model.encounter_probability_ETET = (1 - exp_target)   * fETET * 3   # ETT + E
    model.encounter_probability_TETT = (1 - exp_target)   * fTETT * 3   # ETT + T
    model.encounter_probability_ETE  = (1 - exp_effector) * fETE  * 2   # ET  + E
    model.encounter_probability_ETEE = (1 - exp_effector) * fETEE * 3   # ETE + E
    model.encounter_probability_TETE = (1 - exp_effector) * fTETE * 3   # ETE + T
    return nothing 
end

