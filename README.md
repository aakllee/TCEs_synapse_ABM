# Optimised implementation of Liu et al. 2023 T-cell engagers ABM
9x 1 hour simulations in `reproduce_liu_figure_3` at 2x10^6 cells takes about 60 seconds. Further optimisation may be achieved by using the analytical solution for the binding equations.

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c572d0d4-ff27-4a37-a80e-727ba6dc4951" />


Also included is a reduction of the ABM to ODEs (https://github.com/aakllee/TCEs_synapse_ABM/blob/main/reduction.jl). This is similar to the Liao et al. 2024 model, but more accurately fits the data-points showing the hook effect omitted from the Liao et al. 2024 paper. It more precisely reflects the original ABM, including multiple conjugate formation, and using the original Liu et al. 2023 parameters (currently only `beta` differs, set to 0.028 from Liu et al. 2023's 0.033).

Note `BS3()` is used as our ODEs solver, but this can be trivially changed for greater numerical accuracy if required. 

<img width="300" height="200" alt="liu_ODEs_fig" src="https://github.com/user-attachments/assets/3490f9ec-97ab-4d73-b21c-d670c87ac37d" />


## Usage
### 1. Clone repository
In a terminal, run:
```
$ git clone https://github.com/aakllee/TCEs_synapse_ABM.git
```

### 2. Install Julia
See https://julialang.org.

### 3. Open Julia
For multithreading, specify the number of threads: 
```
$ julia --threads 12
```

### 4. Install dependencies
Enter the package management system (e.g. by typing `]`). Optionally, activate a new project (`pkg> activate .`). Install dependencies:
```
pkg> add Agents, DifferentialEquations, Distributions, Match, PhysicalConstants, Plots, ProgressMeter, Statistics, ThreadsX 
```

### 5. Load files
For the ABM:
```
julia> include("reproduce_liu_figs.jl")
```
Or for the ODEs model:
```
julia> include("reduction.jl")
```

### 6. Run
```
julia> reproduce_liu_fig3()
```

## Authors
**Aaron K. Lee** - Early Oncology DMPK, AstraZeneca, Cambridge, UK

## References
Original paper and model: Liu, C., Zhou, J., Kudlacek, S., Qi, T., Dunlap, T., & Cao, Y. (2023). Population dynamics of immunological synapse formation induced by bispecific T cell engagers predict clinical pharmacodynamics and treatment resistance. eLife, 12. https://doi.org/10.7554/elife.83659

Liao, X., Qi, T., Zhou, J., Liu, C., & Cao, Y. (2024). Optimizing Clinical Translation of Bispecific T‐cell Engagers through Context Unification with a Quantitative Systems Pharmacology Model. Clinical Pharmacology &amp; Therapeutics, 116(2), 415–425. https://doi.org/10.1002/cpt.3302
