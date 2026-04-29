include("liu_agents_main.jl")

function reproduce_liu_fig3(; CD3_mean = 66_299, CD19_mean = 144_866,
                            CD3_geomean = 60_053, CD19_geomean = 130_670)
    liufig3b_data = [0.6487015067924768  1.2831858407079646; 4.887775298393741   2.256637168141593; 19.434038570303596  4.601769911504424; 49.42251490751028   11.283185840707967; 98.54287131604329   12.52212389380531; 196.6991032270088   12.52212389380531; 392.8574323347438   11.858407079646017; 997.1542887677606   0.7079646017699125; 1990.1620644143597  0.8407079646017688]

    mdata = [
        count_free_effectors,  count_free_targets,  count_conjugates,
        count_bound_effectors, count_bound_targets,
        count_all_effectors,   count_all_targets,   count_all_conjugates,
        target_percent_engaged, effector_percent_engaged,
    ]

    TCE_concs = liufig3b_data[:,1]
    t_end = 60 # Minutes
    
    TCR_dist = LogNormal(log(CD3_geomean),  sqrt(2 * (log(CD3_mean)  - log(CD3_geomean))))
    TAA_dist = LogNormal(log(CD19_geomean), sqrt(2 * (log(CD19_mean) - log(CD19_geomean))))
    
    println("Running model...")
    progress_meter = Progress(length(TCE_concs))
    results = ThreadsX.map(
        # For each concentration
        TCE_conc -> begin
            # Run model for 1 hour
            model = initialise_model(; TCE_conc, TCR_dist, TAA_dist)
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
