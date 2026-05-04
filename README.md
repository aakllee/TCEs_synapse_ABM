# Optimised implementation of Liu et al. 2023 T-cell engagers ABM
9x 1 hour simulations in `reproduce_liu_figure_3` at 2x10^6 cells takes about 60 seconds. Further optimisation may be achieved by using the analytical solution for the binding equations.

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c572d0d4-ff27-4a37-a80e-727ba6dc4951" />

## Usage
### 1. Clone repository
In a terminal, run:
```
$ git clone https://github.com/aakllee/TCEs_synapse_ABM.git
```

### 2. Install Julia
See https://julialang.org.

### 3. Install dependencies
Open Julia. Enter the package management system (e.g. by typing `]`). Optionally, activate a new project (`pkg> activate .`). Install dependencies:
```
pkg> add Agents, DifferentialEquations, Distributions, Match, PhysicalConstants, Plots, ProgressMeter. Statistics, ThreadsX 
```

### 4. Load files
```
julia> include("reproduce_liu_figs.jl")
```

### 5. Run code
```
julia> reproduce_liu_fig3()
```

## Authors
**Aaron K. Lee** - Early Oncology DMPK, AstraZeneca, Cambridge, UK

## References
Original paper and model: Can Liu, Jiawei Zhou, Stephan Kudlacek, Timothy Qi, Tyler Dunlap, Yanguang Cao (2023) Population dynamics of immunological synapse formation induced by bispecific T cell engagers predict clinical pharmacodynamics and treatment resistance eLife 12:e83659
