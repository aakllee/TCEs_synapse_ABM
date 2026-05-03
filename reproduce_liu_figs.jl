include("liu_agents_main.jl")

function reproduce_liu_fig3()
    liufig3b_data = [0.6487015067924768  1.2831858407079646; 4.887775298393741   2.256637168141593; 19.434038570303596  4.601769911504424; 49.42251490751028   11.283185840707967; 98.54287131604329   12.52212389380531; 196.6991032270088   12.52212389380531; 392.8574323347438   11.858407079646017; 997.1542887677606   0.7079646017699125; 1990.1620644143597  0.8407079646017688]

    mdata = [
        count_free_effectors,  count_free_targets,  count_conjugates,
        count_bound_effectors, count_bound_targets,
        count_all_effectors,   count_all_targets,   count_all_conjugates,
        target_percent_engaged, effector_percent_engaged,
    ]

    TCE_concs = vcat(liufig3b_data[:,1], 1e4)
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

    percent_effectors_engaged_series = [model_df[end, :effector_percent_engaged] for (_, model_df) in results]
    percent_targets_engaged_series   = [model_df[end, :target_percent_engaged] for (_, model_df) in results]
    
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)
    fig = plot(xlabel = "[Blinatumomab] (ng/ml)", ylabel = "% engaged",
               xscale = :log10, legend = :topleft)

    plot!(fig, TCE_concs, percent_effectors_engaged_series, label = "Sim. Effector", c = 1)
    plot!(fig, TCE_concs, percent_targets_engaged_series,   label = "Sim. Target", c = 2)
    scatter!(fig, liufig3b_data[:,1], liufig3b_data[:,2], label = "Obs. Effector", c = 1)

    display(fig)

    println("Done.")
    return results, percent_effectors_engaged_series, percent_targets_engaged_series, fig
end


function reproduce_liu_fig5(; t_end = 72 * 60, dt = 1)
    mdata = [
        count_free_effectors,  count_free_targets,  count_conjugates,
        count_bound_effectors, count_bound_targets,
        count_all_effectors,   count_all_targets,   count_all_conjugates,
        target_percent_engaged, effector_percent_engaged,
    ]

    TCE_concs = [0.65, 5.0, 20.0, 100.0]

    # Initial cell populations
    n_effector_0 = 1e6
    n_target_0   = 1e6
    
    println("Running model...")
    progress_meter = Progress(length(TCE_concs))
    results = ThreadsX.map(
        # For each concentration
        TCE_conc -> begin
            # Run model for 1 hour
            model = initialise_model(; TCE_conc, n_effector_0, n_target_0, dt)
            time = 0:dt:t_end
            nsteps = Int(t_end / model.dt)
            agent_df, model_df = run!(model, nsteps; mdata, showprogress = false);
            next!(progress_meter)
            (agent_df, model_df)
        end, TCE_concs)
    finish!(progress_meter)

    times = [model_df[:, :time] for (_,model_df) in results] .* dt ./60
    target_cell_depletions = [(1 .- model_df[:, :count_free_targets] ./ n_target_0) .* 100
                              for (_,model_df) in results]
    println("Done.")
    println("Plotting...")
    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)
    fig = plot(xlabel = "Time (hours)", ylabel = "Target cell depletion (%)", legend = :topleft)
    for (i, target_cell_depletion_series) in enumerate(target_cell_depletions)
        plot!(fig, times, target_cell_depletion_series; label = "$(TCE_concs[i]) ng/ml")
    end
    display(fig)
    println("Done.")
    return (results, target_cell_depletions)
end
