# RingAttractorMovement
This project provides the code to explore how different animal movement trajectories arise from the underlying dynamics of neural ring attractors [1]. It is instructive to first read [1] to understand this project. The code was written in Python.

## Code structure
### Pattern.py
This file contains the lower-level objects and functions that are called by the rest of the files. Specifically, the following objects are included:

- BehProp: saves the parameters of the neural dynamics and animal movement (define animal behaviour).
- IndProp: saves multiple different behaviours of a single individual (a list of BehProp objects) as well as a specification of stochastic transition rates between them.
- EnvProp: saves the parameters of the environment.
- Pattern: based on a list of individuals (IndProp objects) and environmental properties (EnvProp), this object allows for stochastic and mean-field simulations of the resulting dynamics - see [1] for details.
- TwoTargets: used for analytical predictions in Fig. S5 of [1]
  
Additionally, multiple plotting functions that use these objects are included. For details, see the documentation of each object and plotting function.

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
