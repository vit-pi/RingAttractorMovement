# RingAttractorMovement
This project provides the code to explore how different animal movement trajectories arise from the underlying dynamics of neural ring attractors [1]. It is instructive to first read [1] to understand this project. The code was written in Python.

## Code structure
### Pattern.py
This file contains the lower-level objects and functions that are called by the rest of the files. Objects EnvProp and SpecProp store information about the information of the environment where the studied organisms move (EnvProp) and about the properties of these organisms (SpecProp). The IntFit objects carry the information about the fecundity functions of each organism (see [1]) and are passed as an attribute to the EnvProp object. The EnvProp and SpecProp objects are passed to the Pattern object (the main object), which holds the information about the spatial distribution of organisms of all diffusivities at a particular time. This object contains methods to (1) transform between data structures that describe this distribution, (2) find the properties of this distribution, and (3) update this distribution in time.  The important update functions are:

1) update_pattern_euler(): used to update the distribution and time $t \to t + \Delta t$,
2) mutation_update(): used before every time-step of the Euler method above if mutants of variable diffusivity are to be introduced into the population,
3) fitness_update(): used before every time-step of the Euler method and before mutation_update() to keep track of the total fitness of each organism,
4) fitness_update(): used before every time-step of the Euler method and before mutation_update() to keep track of the expected diffusivity of each organism.

To initialize the Pattern object, the initial pattern is required in addition to EnvProp and SpecProp objects. This can be achieved with the init_pattern function. To make a desired plot of the current state of the Pattern object, the plotting functions are used.

### Fig_X.py
These files reproduce various figure panels X in [1]. If more than one panel is provided (e.g. Fig_X_Y.py), the beginning of the file includes a boolean variable (usually called deterministic, corresponding to mean-field as opposed to stochastic simulations) and a description how to set this variable (True/False) to reproduce an appropriate figure panel (X/Y). The output preliminary figures are saved in the Figures folder.

## Installation guide
(Tested on macOS Sequoia 15.6.1, Apple M4 Max, 128 GB RAM, Python and PyCharm installed. The estimated run time for producing a movement trajectory in [1]: 5 minutes (where the exact runt-time might depending on a particular task.)
1) Install Python (https://www.python.org/) and PyCharm (https://www.jetbrains.com/pycharm/).
2) Download this project and save it into a folder.
3) Open this folder with PyCharm.

## Guide to reproduce Figures from [1]
1) Open a file of interest (Fig_X.py based on the table above).
2) Modify the boolean variable if needed (see instructions in each file).
3) Run the file.
4) Output plot(s) appear in the Figures folder.

## Demo
1) Open Fig_1.py.
2) No parameters need to be modified in this case.
3) Run the file.
4) Open the folder "Figure" to find the new files Fig_ModelOverview.png and Fig_ModelOverview.svg.

# References
[1] V. Piskovsky, A. Paula, D. Gorbonos, P.K. Maini, I.D. Couzin, The neural basis of spatial exploration and decision-making, 2025.
