include("reduction.jl")

using Plots
using ThreadsX

"Example usage: plot(reproduce_liu_fig3()..., size=(1200, 1200))"
function reproduce_liu_fig3()
    liufig3b_data = [0.6487015067924768  1.2831858407079646; 4.887775298393741   2.256637168141593; 19.434038570303596  4.601769911504424; 49.42251490751028   11.283185840707967; 98.54287131604329   12.52212389380531; 196.6991032270088   12.52212389380531; 392.8574323347438   11.858407079646017; 997.1542887677606   0.7079646017699125; 1990.1620644143597  0.8407079646017688]

    default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)

    ### Concentration -- Effector % engaged ###
    fig1 = plot(; xlabel = "[Blinatumomab], ng/mL", ylabel = "Effector % engaged", xscale = :log10, ylims = (0, 20))    
    TCE_concs = liufig3b_data[:,1]
    ctoxs = liufig3b_data[:,2]
    TCE_concs_sim = 10 .^ range(log10(minimum(TCE_concs))-1, log10(maximum(TCE_concs))+1, 50)

    # Solve model for each concentration
    sols = ThreadsX.map(TCE_conc -> run_model(60; TCE_conc), TCE_concs_sim)
    synapse_effector = map(get_effector_percent_engaged, sols)
    scatter!(fig1, TCE_concs, ctoxs; c = 1, label = "Obs.")
    plot!(fig1, TCE_concs_sim, synapse_effector; c = 1, label = "Sim.")


    ### Incubation -- Effector % engaged ### 
    fig2 = plot(; xlabel = "Incubation (min)", ylabel = "Effector % engaged", ylims = (0, 20))
    times = [0, 15, 30, 45, 60]
    TCE_concs = [20, 100]
    for (c, TCE_conc) in enumerate(TCE_concs)
        sols = ThreadsX.map(time -> run_model(time; TCE_conc), times)
        synapse_effector = map(get_effector_percent_engaged, sols)
        plot!(fig2, times, synapse_effector; c, label = "$(TCE_conc) ng/mL")
    end

    ### CD3 expression -- Effector % engaged ###
    fig3 = plot(; xlabel = "CD3 expression", ylabel = "Effector % engaged", ylims = (0, 50))
    CD3_expressions = [66_299, 126_067, 186_927]
    TCE_concs = [5, 20, 100]
    for (c, TCE_conc) in enumerate(TCE_concs)
        sols = ThreadsX.map(TCR_dist -> run_model(60; TCE_conc, TCR_dist), CD3_expressions)
        synapse_effector = map(get_effector_percent_engaged, sols)
        plot!(fig3, CD3_expressions, synapse_effector; c, label = "$(TCE_conc) ng/mL")
    end

    ### CD19 expression -- Effector % engaged ###
    fig4 = plot(; xlabel = "CD19 expression", ylabel = "Effector % engaged", ylims = (0, 30))
    CD19_expressions = let L = 66_186, M = 144_866, H = 189_920
        [L/10, L, M, H, M*2, M*4]
    end
    TCE_concs = [5, 20, 100]
    for (c, TCE_conc) in enumerate(TCE_concs)
        sols = ThreadsX.map(TAA_dist -> run_model(60; TCE_conc, TAA_dist), CD19_expressions)
        synapse_effector = map(get_effector_percent_engaged, sols)
        plot!(fig4, CD19_expressions, synapse_effector; c, label = "$(TCE_conc) ng/mL")
    end

    ### Cell density -- Effector % engaged, CD3 L, CD19 M ###
    fig5 = plot(; xlabel = "Cell density", ylabel = "Effector % engaged", ylims = (0, 50))
    cell_densities = 1e6 .* [0.7, 2, 3, 4, 6, 8]
    TCE_concs = [5, 20, 100]
    TCR_dist = 66_299
    TAA_dist = 144_866
    for (c, TCE_conc) in enumerate(TCE_concs)
        sols = ThreadsX.map(
            cell_density -> begin
                n_effector_0 = cell_density / 2
                n_target_0 = cell_density / 2
                run_model(60; TCE_conc, TCR_dist, TAA_dist, n_effector_0, n_target_0)
            end , cell_densities)
        synapse_effector = map(get_effector_percent_engaged, sols)
        plot!(fig5, cell_densities, synapse_effector; c, label = "$(TCE_conc) ng/mL")
    end

    ### Cell density -- Effector % engaged, CD3 H, CD19 M ###
    fig6 = plot(; xlabel = "Cell density", ylabel = "Effector % engaged", ylims = (0, 50))
    cell_densities = 1e6 .* [0.7, 2, 3, 4, 6, 8]
    TCE_concs = [5, 20]
    TCR_dist = 186_927
    TAA_dist = 144_866
    for (c, TCE_conc) in enumerate(TCE_concs)
        sols = ThreadsX.map(
            cell_density -> begin
                n_effector_0 = cell_density / 2
                n_target_0 = cell_density / 2
                run_model(60; TCE_conc, TCR_dist, TAA_dist, n_effector_0, n_target_0)
            end , cell_densities)
        synapse_effector = map(get_effector_percent_engaged, sols)
        plot!(fig6, cell_densities, synapse_effector; c, label = "$(TCE_conc) ng/mL")
    end

    ### E:T % engaged, 100 ng/mL ###
    fig7 = plot(; xlabel = "E:T", ylabel = "E or T cell % engaged", ylims = (0, 50))
    ET_ratios = [1/20, 1/10, 1/5, 1/3, 1/2, 1, 2, 3, 5, 10, 20]
    ET_ratios_labs = ["1:20", "1:10", "1:5", "1:3", "1:2", "1:1", "2:1", "3:1", "5:1", "10:1", "20:1"]
    TCE_conc = 100
    sols = ThreadsX.map(
        ET_ratio -> begin
            n_effector_0 = 4e6 * ET_ratio / (ET_ratio + 1)
            n_target_0   = 4e6 *        1 / (ET_ratio + 1)
            run_model(60; TCE_conc, n_effector_0, n_target_0)
        end, ET_ratios)
    synapse_effector = map(get_effector_percent_engaged, sols)
    synapse_target   = map(get_target_percent_engaged, sols)
    plot!(fig7, ET_ratios_labs, synapse_effector; label = "Effector cell %")
    plot!(fig7, ET_ratios_labs, synapse_target; label = "Target cell %")

    ### E:T % engaged, 20 ng/mL ###
    fig8 = plot(; xlabel = "E:T", ylabel = "E or T cell % engaged", ylims = (0, 50))
    ET_ratios = [1/20, 1/10, 1/5, 1/3, 1/2, 1, 2, 3, 5, 10, 20]
    ET_ratios_labs = ["1:20", "1:10", "1:5", "1:3", "1:2", "1:1", "2:1", "3:1", "5:1", "10:1", "20:1"]
    TCE_conc = 20
    sols = ThreadsX.map(
        ET_ratio -> begin
            n_effector_0 = 4e6 * ET_ratio / (ET_ratio + 1)
            n_target_0   = 4e6 *        1 / (ET_ratio + 1)
            run_model(60; TCE_conc, n_effector_0, n_target_0)
        end, ET_ratios)
    synapse_effector = map(get_effector_percent_engaged, sols)
    synapse_target   = map(get_target_percent_engaged, sols)
    plot!(fig8, ET_ratios_labs, synapse_effector; label = "Effector cell %")
    plot!(fig8, ET_ratios_labs, synapse_target; label = "Target cell %")
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8
end
