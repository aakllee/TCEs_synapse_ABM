# Implementation developed by Aaron K. Lee.
# Early Oncology DMPK, AstraZeneca, Cambridge, UK.
# Department of Computer Science, University of Warwick, Coventry, UK.

# Original paper and model: Can Liu, Jiawei Zhou, Stephan Kudlacek,
# Timothy Qi, Tyler Dunlap, Yanguang Cao (2023) Population dynamics of
# immunological synapse formation induced by bispecific T cell
# engagers predict clinical pharmacodynamics and treatment resistance
# eLife 12:e83659

# This one is attempting to replicate the model with Agents.jl.
# Much more performant, readable, and robust.
# Simulation of 9x 1 hour at 2x10^6 cells takes about 60 seconds.
# Further optimisation may be achieved by using the analytical
# solution for the binding equations.

using Agents
using DifferentialEquations
using Distributions
using Match
using PhysicalConstants
using Plots
using ProgressMeter
using Statistics
using ThreadsX
using Unzip

include("liu_agents_liu_functions.jl")

@kwdef mutable struct Parameters{T<:Real, DistT<:Distribution}
    n_effector_0::T  = 1e6 # Initial effector cells, 1/mL.
    n_target_0::T    = 1e6 # Initial target cells, 1/mL.
    
    # Simulation parameters
    Na::T = PhysicalConstants.CODATA2022.N_A.val # Avogradro constant, 1/mol
    dt::Int64 = 1 # Timestep, mins.

    # Model parameters
    TCE_conc::T    = 1.0     # [TCE], flag_nM ? nM : ng/mL.
    flag_nM::Bool  = false
    MW::T          = 54_100  # TCE molecular weight, g/mol. 
    TCR_dist::DistT = Dirac{T}(66_299)  # TCR antigen distribution, 1/cell.
    TAA_dist::DistT = Dirac{T}(144_866) # TAA antigen distribution, 1/cell.
    tau_synapse::T = 150     # Synapse duration (in vitro model), mins.
    KD_TCR::T      = 2.6e-7  # TCR binding affinity, M. 
    KD_TAA::T      = 1.49e-9 # TAA binding affinity, M.
    S_effector::T  = 4*pi * (5.0^2) * 1.8 # Effector surface area, um^2.
    S_target::T    = 4*pi * (6.0^2) * 1.8 # Target surface area, um^2.
    D::T           = 0.83  # Cell diffusion coefficient, um^2/s.
    R_system::T    = 6200  # Spherical diameter of reaction system, um.
    R_target::T    = 6.0   # Radius of target cell, um.
    R_effector::T  = 5.0   # Radius of effector cell, um.
    kint::T        = 0.002 # Internalisation rate of TAA, unitless.
    beta::T        = 0.033 # Binding constant, unitless.
    Sc1::T         = 5.0   # Synapse surface contact area, um^2. 
    
    # Spatial constants
    fETE::T        = 0.75
    fTET::T        = 0.66
    fETET::T       = 0.75
    fETEE::T       = 0.5
    fTETT::T       = 0.33
    fTETE::T       = 0.67

    # Variables
    n_effector_free::Int64     = 1e6 # Free effector cell population, cells/mL.
    n_target_free::Int64       = 1e6 # Free target cell population, cells/mL.
    n_conjugate::Matrix{Int64} = zeros(Int64, 3, 3) # Conjugate population, cells/mL.
    equilibrium_TCR_free::T    = 0.0 # Free TCR at equilibrium, 1/um^2.
    equilibrium_TCR_binary::T  = 0.0 # TCR binary complex at equilibrium, 1/um^2.
    equilibrium_TAA_free::T    = 0.0 # Free TAA at equilibrium, 1/um^2.
    equilibrium_TAA_binary::T  = 0.0 # TAA binary complex at equilibrium, 1/um^2.
    konA::T  = NaN # To be calculated by calculate_rate_constants
    konB::T  = NaN # To be calculated by calculate_rate_constants
    koffA::T = NaN # To be calculated by calculate_rate_constants
    koffB::T = NaN # To be calculated by calculate_rate_constants
    encounter_probability_ET::T   = 0.0
    encounter_probability_TET::T  = 0.0
    encounter_probability_ETET::T = 0.0
    encounter_probability_TETT::T = 0.0
    encounter_probability_ETE::T  = 0.0
    encounter_probability_ETEE::T = 0.0
    encounter_probability_TETE::T = 0.0
    cache::ODECache{T} = create_ode_cache((; konA, konB, koffA, koffB), T)
end

# Free effector.
@agent struct Effector(NoSpaceAgent)
    TCR_0::Int64        # Assigned TCR antigen number on cell (e.g. CD3), molecules per cell.
    TCR_free::Float64   # Free TCR on cell surface, /um^2.
    TCR_binary::Float64 # TCR in binary complex on cell surface, /um^2.
    
    TCR_free_0::Float64
    TCR_binary_0::Float64
end

# Free target.
@agent struct Target(NoSpaceAgent)
    TAA_0::Int64        # TAA antigen number on cell (e.g. CD19), molecules per cell.
    TAA_free::Float64   # Free TAA on cell surface, /um^2.
    TAA_binary::Float64 # TAA in binary complex on cell surface, /um^2. 
end

# Conjugate of N effectors, M targets.
@agent struct Conjugate(NoSpaceAgent)
    time_formed::Float64
    effectors::Vector{Effector}
    targets::Vector{Target}
end

# Define Cells type as any agent in the system 
@multiagent Cells(Effector, Target, Conjugate) <: AbstractAgent


########### Utilities ###########

"Returns a vector of cells of type cell_type in model."
get_cells(cell_type, model) = [cells for cells in allagents(model) if variantof(cells) == cell_type]

"Counts number of cells of type cell_type in model."
function count(cell_type, model)
    # @warn "Generic count(cell_type, model) is slow. Use count_free_effectors, count_free_targets, count_conjugates instead."
    # return count(cell -> variantof(cell) == cell_type, allagents(model))
    if cell_type == Effector
        return count_free_effectors(model)
    elseif cell_type == Target
        return count_free_targets(model)
    elseif cell_type == Conjugate
        return count_all_conjugates(model)
    end
end

count_free_effectors(model) = model.n_effector_free
count_free_targets(model)   = model.n_target_free
count_all_conjugates(model) = sum(model.n_conjugate)
count_conjugates(model)     = copy(model.n_conjugate)

count_bound_effectors(model) = sum([n_E * count_conjugates(n_E, n_T, model) for n_E in 1:3, n_T in 1:3])
count_bound_targets(model)   = sum([n_T * count_conjugates(n_E, n_T, model) for n_E in 1:3, n_T in 1:3])

"Counts conjugates with n_effectors effectors, and n_targets targets in model."
count_conjugates(n_effectors, n_targets, model) = model.n_conjugate[n_effectors, n_targets]

"Counts total free effectors and those in conjugates."
count_all_effectors(model) = count_free_effectors(model) + count_bound_effectors(model)

"Counts total free targets and those in conjugates."
count_all_targets(model) = count_free_targets(model) + count_bound_targets(model)

"Checks if there exists any cell of variant cell_type in model."
exists_alive_effector(model)  = count_free_effectors(model) > 0 
exists_alive_target(model)    = count_free_targets(model)   > 0
exists_alive_conjugate(model) = count_all_conjugates(model) > 0
exists_alive(cell_type, model) = if cell_type == Effector; exists_alive_effector(model)
elseif cell_type == Target;    exists_alive_target(model)
elseif cell_type == Conjugate; exists_alive_conjugate(model)
else; throw("Invalid cell type $(cell_type) in exists_alive.")
end

"Returns a random cell, or nothing if none exist/alive."
function get_random_cell(cell_type, model; reservoir_threshold = 0.0001)
    !exists_alive(cell_type, model) && return nothing
    variant_is_a_cell_type = get_isa_fn(cell_type)
    return random_agent(model, variant_is_a_cell_type)
end

is_effector(cells::Cells)  = variantof(cells) == Effector
is_target(cells::Cells)    = variantof(cells) == Target
is_conjugate(cells::Cells) = variantof(cells) == Conjugate
function get_isa_fn(cell_type)
    cell_type == Effector  && return is_effector
    cell_type == Target    && return is_target
    cell_type == Conjugate && return is_conjugate
    return _ -> false
end

get_random_target(model) = get_random_cell(Target, model)
get_random_effector(model) = get_random_cell(Effector, model)

target_percent_engaged(model)   = 100 * count_bound_targets(model)   / count_all_targets(model)
effector_percent_engaged(model) = 100 * count_bound_effectors(model) / count_all_effectors(model)

# Supposedly, "CD3 downregulation is unchanged after 1 hour".
# Does this mean we can simply cap t to 60?
# Or does it mean that TCR_hill(TCR_binary(t), t) for
# t > 60 = TCR_hill(TCR_binary(60), 60)?
# i.e. do we fix the effective TCR_binary too?
# Wouldn't make much sense but could be what they do...

# I'd say we can fix TCR_free after t = 60 instead. TODO.
# Only relevant for in vitro and in vivo models, which are
# not provided by the Liu et al. paper. 

@fastmath TCR_hill(TCR_binary, t; gamma = 0.9, h = 0.7) = 1 / (1 + 0.1 * TCR_binary^gamma * min(t, 60)^h)
@fastmath function TCR_downregulation!(effector, model)
    ratio = TCR_hill(effector.TCR_binary_0, abmtime(model))
    effector.TCR_free   = effector.TCR_free_0   * ratio
    effector.TCR_binary = effector.TCR_binary_0 * ratio

    return nothing
end
@fastmath function TAA_internalisation!(target, model)
    target.TAA_binary *= exp(-model.kint * model.dt)
    return nothing
end

"Returns free and binary antigen concentrations /um^2 on surface."
@fastmath function ag_equilibrium(ag_0, ag_dist, eqm_free, eqm_binary)
    ag_frac   = float(ag_0) / float(mean(ag_dist))
    ag_free   = ag_frac * eqm_free   # Cell surface free TCR /um^2
    ag_binary = ag_frac * eqm_binary # Cell surface TCR binary complex /um^2
    return (; ag_free, ag_binary)
end
"Sets surface antigen values proportional to equilibrium."
function TCR_equilibriate!(effector, model)
    (; ag_free, ag_binary) = ag_equilibrium(effector.TCR_0, model.TCR_dist, model.equilibrium_TCR_free, model.equilibrium_TCR_binary)
    effector.TCR_free   = ag_free
    effector.TCR_binary = ag_binary
    return nothing
end
function TAA_equilibriate!(target, model)
    (; ag_free, ag_binary) = ag_equilibrium(target.TAA_0, model.TAA_dist, model.equilibrium_TAA_free, model.equilibrium_TAA_binary)
    target.TAA_free   = ag_free
    target.TAA_binary = ag_binary
    return nothing
end

ngmL_to_M(conc, MW) = conc * 1000 * 1e-9 / MW


# !! Note that the order of adding cells and updating counts is
# important in the below binding functions. It may be clearer and
# easier to trace to explicitly track old vs new cell numbers, but
# this is fine for now. Just check that total target and effector
# numbers are conserved. !!

"Forms a conjugate of an effector and target in the model. This
removes the targets and effectors from the simulation until the
synapse is dissolved."
function form_conjugate!(effector, target, model)
    add_agent!(Cells∘Conjugate, model; time_formed = abmtime(model),
               effectors = Effector[variant(effector)],
               targets   = Target[variant(target)])
    remove_agent!(effector, model)
    remove_agent!(target,   model)

    # Update free cell and conjugate counts. 
    model.n_target_free   -= 1
    model.n_effector_free -= 1
    model.n_conjugate[1,1] += 1
    return nothing
end
function bind_conjugate!(effector::Effector, conjugate, model)
    model.n_conjugate[length(conjugate.effectors), length(conjugate.targets)] -= 1

    push!(conjugate.effectors, effector) # Add effector to conjugate.
    remove_agent!(effector, model)       # Remove effector from system.

    model.n_effector_free -= 1
    model.n_conjugate[length(conjugate.effectors), length(conjugate.targets)] += 1
    return nothing
end
function bind_conjugate!(target::Target, conjugate, model)
    model.n_conjugate[length(conjugate.effectors), length(conjugate.targets)] -= 1
    
    push!(conjugate.targets, target) # Add target to conjugate.
    remove_agent!(target, model)     # Remove target from system.

    model.n_target_free -= 1
    model.n_conjugate[length(conjugate.effectors), length(conjugate.targets)] += 1
    return nothing
end
"Removes conjugate, and returns effectors to free pool.
Target cells are NOT returned, as they are considered destroyed."
function dissolve_conjugate!(conjugate, model)
    # Update conjugate and free effector counts.
    model.n_conjugate[length(conjugate.effectors), length(conjugate.targets)] -= 1
    model.n_effector_free += length(conjugate.effectors)

    add_agent!.(Cells.(conjugate.effectors), Ref(model)) # Return all effectors to free pool.
    remove_agent!(conjugate, model) # Remove conjugate from model. 
    return nothing
end

check_encounter(cells, model) = check_encounter(cells, model, variant(cells))

"Returns vector of cells to bind. Empty vector if none."
function check_encounter(conjugate, model, ::Conjugate)
    num_effectors = length(conjugate.effectors)
    num_targets   = length(conjugate.targets)

    x_encounter = rand(abmrng(model)) # Encounter?

    # (N_E, N_T) => (P(bind_E), P(bind_T))
    (P_bind_E, P_bind_T) = @match (num_effectors, num_targets) begin
        (1, 1) => (model.encounter_probability_ETE,  model.encounter_probability_TET)
        (1, 2) => (model.encounter_probability_ETET, model.encounter_probability_TETT)
        (2, 1) => (model.encounter_probability_ETEE, model.encounter_probability_TETE)
        # Conjugates with E or T > 2 do not form higher order conjugates.
        _ => return [] 
    end
    to_bind_or_not_to_bind = (
        E =            x_encounter <= P_bind_E            && exists_alive_effector(model),
        T = P_bind_E < x_encounter <= P_bind_E + P_bind_T && exists_alive_target(model),
    )

    # Return vector of cell(s) to bind.
    return if to_bind_or_not_to_bind.T && to_bind_or_not_to_bind.E
        [get_random_effector(model), get_random_target(model)]
    elseif to_bind_or_not_to_bind.T
        [get_random_target(model)]
    elseif to_bind_or_not_to_bind.E
        [get_random_effector(model)]
    else
        []
    end
end

# Returns target to bind, or nothing if probability too low/no free target exists.
function check_encounter(effector, model, ::Effector)
    x_encounter = rand(abmrng(model)) # Encounter?
    x_encounter > model.encounter_probability_ET && return nothing # Exit if no encounter.
    !exists_alive_target(model) && return nothing # Exit if no targets available.
    return get_random_target(model) # Return a random free target.
end

###################################

"Agent step function. Branches to appropriate step function for cell
variant type. You should leave this one well alone."
agent_step!(agent, model) = agent_step!(agent, model, variant(agent))

"Effector step function. Runs for each effector on each timestep."
function agent_step!(effector, model, ::Effector)
    # TCR_equilibriate!(effector, model)
    TCR_downregulation!(effector, model)

    # Encounter target. 
    target_encountered = check_encounter(effector, model)
    isnothing(target_encountered) && return nothing # Next effector

    # Calculate binding (adhesion) probability.
    binding_probability = calculate_binding_probability!(effector, target_encountered, model)

    # Quit if no binding.
    rand(abmrng(model)) > binding_probability && return nothing # Next effector

    # If binding success, form conjugate. 
    form_conjugate!(effector, target_encountered, model)
    return nothing
end

"Target step function. Runs for each target on each timestep."
function agent_step!(target, model, ::Target)
    # TAA_equilibriate!(target, model)
    TAA_internalisation!(target, model)
end

"Conjugate step function. Runs for each conjugate on each timestep."
function agent_step!(conjugate, model, ::Conjugate)
    # abmtime(model) == conjugate.time_formed && return nothing # Quit if same timestep as formation
    
    # Renew antigen densities for child effectors and targets
    @inbounds for effector in conjugate.effectors
        # TCR_equilibriate!(effector, model)
        TCR_downregulation!(effector, model)
    end
    @inbounds for target in conjugate.targets
        # TAA_equilibriate!(target, model)
        TAA_internalisation!(target, model)
    end

    # If tau_synapse has elapsed, kill target(s) and return
    # effector(s) to free pool.
    # Should dissolution be per bound E-T pair? i.e. need to record
    # time of each binding rather than just 1,1 formation? Then
    # dissociate each bond as tau elapses for each?
    if abmtime(model) > conjugate.time_formed + model.tau_synapse
        dissolve_conjugate!(conjugate, model)
        return nothing
    end

    # Use encounter and binding probabilities to bind free effectors
    # and targets to conjugate.
    cells_to_bind = check_encounter(conjugate, model)
    # isnothing(cells_to_bind) && return nothing
    for cell_to_bind in cells_to_bind
        # isnothing(cell_to_bind) && return nothing # Exit if nothing to bind
        binding_probability = if variantof(cell_to_bind) == Effector
            # Select a random target in conjugate to bind free effector to
            target = rand(abmrng(model), conjugate.targets)
            calculate_binding_probability!(cell_to_bind, target, model)
        elseif variantof(cell_to_bind) == Target
            # Select a random effector in conjugate to bind free target to
            effector = rand(abmrng(model), conjugate.effectors)
            calculate_binding_probability!(effector, cell_to_bind, model)
        end
        
        x_binding = rand(abmrng(model)) # Binding?
        x_binding > binding_probability && return nothing # Quit if no binding.
        bind_conjugate!(variant(cell_to_bind), conjugate, model) # Bind to conjugate
    end
    
    return nothing
end

function model_step!(model)
    # Calculate encounter probabilities
    calculate_encounter_probabilities!(model)
end

function initialise_model(;
                          n_effector_0  = 1e6, # Initial effector cells, 1/mL.
                          n_target_0    = 1e6, # Initial target cells, 1/mL.
                          
                          # Simulation parameters
                          Na = PhysicalConstants.CODATA2022.N_A.val, # Avogradro constant, 1/mol
                          dt = 1, # Timestep, mins.
                          # Model parameters
                          TCE_conc      = 1.0,     # [TCE], flag_nM ? nM : ng/mL.
                          flag_nM       = false,
                          MW            = 54_100,  # TCE molecular weight, g/mol. 
                          TCR_dist      = 66_299,  # TCR antigen distribution, 1/cell. <:Real for uniform, <:Distribution for distribution, e.g. LogNormal(...).
                          TAA_dist      = 144_866, # TAA antigen distribution, 1/cell. <:Real for uniform, <:Distribution for distribution, e.g. LogNormal(...).
                          tau_synapse   = 150,     # Synapse duration (in vitro model), mins.
                          KD_TCR        = 2.6e-7,  # TCR binding affinity, M. 
                          KD_TAA        = 1.49e-9, # TAA binding affinity, M.
                          S_effector    = 4*pi * (5.0^2) * 1.8, # Effector surface area, um^2.
                          S_target      = 4*pi * (6.0^2) * 1.8, # Target surface area, um^2.
                          D             = 0.83, # Cell diffusion coefficient, um^2/s.
                          R_system      = 6200, # Spherical diameter of reaction system, um.
                          R_target      = 6.0,  # Radius of target cell, um.
                          R_effector    = 5.0,  # Radius of effector cell, um.
                          fETE  = 0.75, fTET  = 0.66, fETET = 0.75,
                          fETEE = 0.5,  fTETT = 0.33, fTETE = 0.67,
                          kint = 0.002, beta = 0.033,
                          Sc1 = 5.0,)
    @assert(typeof(TCR_dist) == typeof(TAA_dist) && TCR_dist isa Real && TAA_dist isa Real,
            "TODO: distributions for TCR and TAA")
    
    properties = Parameters{Float64, Dirac}(
        # Simulation parameters
        ; Na, dt,
        # Model parameters
        TCE_conc, flag_nM, MW,
        n_effector_0, n_target_0,
        TCR_dist = Dirac{Float64}(TCR_dist),
        TAA_dist = Dirac{Float64}(TAA_dist),
        KD_TCR, KD_TAA,
        S_effector, S_target,
        tau_synapse,
        D, R_system, R_target, R_effector,
        fETE, fTET, fETET, fETEE, fTETT, fTETE,
        kint, beta, Sc1, 
        calculate_rate_constants(KD_TCR, KD_TAA; Na)..., # Set calculated rate constants
        
        # Variables
        n_effector_free = n_effector_0, # Set initial effector cell population
        n_target_free   = n_target_0,   # Set initial target cell population
        # n_conjugate = zeros(Int64,3,3),
    )

    # Create model
    model = StandardABM(Cells; agent_step!, model_step!, properties)

    # Calculate antigen density at equilibrium.
    calculate_equilibrium!(model) # molecules/um^2
    
    # Initialise effector cell population
    @inbounds for _ in 1:n_effector_0
        # Assign TCR value from distribution ([A]', /cell), calculate
        # equilibrium [A] and [AY], /um^2, add effector cell to model.
        TCR_0 = typeof(TCR_dist)<:Real ? TCR_dist : rand(abmrng(model), TCR_dist)
        (; ag_free, ag_binary) = ag_equilibrium(TCR_0, TCR_dist, model.equilibrium_TCR_free, model.equilibrium_TCR_binary)
        add_agent!(Cells ∘ Effector, model; TCR_0, TCR_free = ag_free, TCR_binary = ag_binary,
                   TCR_free_0 = ag_free, TCR_binary_0 = ag_binary)
    end

    # Initialise target cell population
    @inbounds for _ in 1:n_target_0
        # Assign TAA value from distribution ([B]', /cell), calculate
        # equilibrium [B] and [YB], /um^2, add effector cell to model.
        TAA_0 = typeof(TAA_dist)<:Real ? TAA_dist : rand(abmrng(model), TAA_dist)
        (; ag_free, ag_binary) = ag_equilibrium(TAA_0, TAA_dist, model.equilibrium_TAA_free, model.equilibrium_TAA_binary)
        add_agent!(Cells ∘ Target, model; TAA_0, TAA_free = ag_free, TAA_binary = ag_binary)
    end
    
    return model
end

function run_liu_ABM!(; t_end = 60, showprogress = true, kwargs...)
    model = initialise_model(; kwargs...)
    nsteps = Int(t_end / model.dt)

    mean_TCR(model) = (; free = mean([effector.TCR_free   for effector in get_cells(Effector, model)]),
                       binary = mean([effector.TCR_binary for effector in get_cells(Effector, model)]))
    mean_TAA(model) = (; free = mean([target.TAA_free     for target   in get_cells(Target,   model)]),
                       binary = mean([target.TAA_binary   for target   in get_cells(Target,   model)]))
    
    mdata = [
        count_free_effectors,  count_free_targets,  count_conjugates,
        count_bound_effectors, count_bound_targets,
        count_all_effectors,   count_all_targets,   count_all_conjugates,
        target_percent_engaged, effector_percent_engaged,
        mean_TCR, mean_TAA,
    ]
    
    agent_df, model_df = run!(model, nsteps; mdata, showprogress);

    time = 0:model.dt:t_end

    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)
    fig = plot(xlabel = "Time (minutes)", ylabel = "Number (per ml)")

    plot!(fig, model_df[!, :count_free_effectors], label = "Free effectors")
    plot!(fig, model_df[!, :count_free_targets],   label = "Free targets")
    plot!(fig, model_df[!, :count_all_conjugates], label = "Conjugates")

    display(fig)
    
    return model, agent_df, model_df, fig
end

function reproduce_liu_fig3()
    liufig3b_data = [0.6487015067924768  1.2831858407079646; 4.887775298393741   2.256637168141593; 19.434038570303596  4.601769911504424; 49.42251490751028   11.283185840707967; 98.54287131604329   12.52212389380531; 196.6991032270088   12.52212389380531; 392.8574323347438   11.858407079646017; 997.1542887677606   0.7079646017699125; 1990.1620644143597  0.8407079646017688]

    mdata = [
        count_free_effectors,  count_free_targets,  count_conjugates,
        count_bound_effectors, count_bound_targets,
        count_all_effectors,   count_all_targets,   count_all_conjugates,
        target_percent_engaged, effector_percent_engaged,
    ]

    TCE_concs = liufig3b_data[:,1]
    t_end = 60 # Minutes

    println("Running model...")
    progress_meter = Progress(length(TCE_concs))
    results = ThreadsX.map(
        # For each concentration
        TCE_conc -> begin
            # Run model for 1 hour
            model = initialise_model(; TCE_conc)
            time = 0:model.dt:t_end
            nsteps = Int(t_end / model.dt)
            agent_df, model_df = run!(model, nsteps; mdata, showprogress = false);
            next!(progress_meter)
            (agent_df, model_df)
        end, TCE_concs)
    finish!(progress_meter)

    println("Done.")
    println("Plotting...")

    pIS_values = [model_df[end, :target_percent_engaged] for (_, model_df) in results]
    
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)
    fig = plot(xlabel = "[Blinatumomab] (ng/ml)", ylabel = "Effector % engaged",
               xscale = :log10, legend = :topleft)

    plot!(fig, TCE_concs, pIS_values, label = "Sim.", c = 1)
    scatter!(fig, liufig3b_data[:,1], liufig3b_data[:,2], label = "Obs.", c = 1)

    display(fig)

    println("Done.")
    return results, pIS_values, fig
end
