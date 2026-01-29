# Ring Attractor Spin Model (Figure S4 reproduction)

Python code to rerun the Figure S4 simulations and regenerate panels S4C, S4F, S4J, S4G, and S4K.
This repo is a minimal, reproducible subset of a larger project that can also generate a wider range of stochastic and systematic search patterns and decision behaviors driven by a neural ring-attractor model.
## Structure
- `core/` contains the model implementation:
  - `core/simulation.py` orchestrates agent + environment + spin dynamics.
  - `core/spinSystem.py`, `core/perceptionSystem.py`, `core/environment.py`, `core/agent.py` implement components.
  - `core/experiment_utils.py` provides HDF5 utilities and parameter logging.
- `Fig_S4CFJ.py` runs the angle sweeps and plots S4C/S4F/S4J.
- `Fig_S4GK.py` runs fixed-angle sweeps (kappa/strength) and plots S4G/S4K.
- `figureS4*.json` are the configs for each panel.

## Installation
1. Create a conda env:
   ```
   conda env create -f environment.yml
   conda activate spinmodel
   ```

## Run all Figure S4 panels
From this folder:
```
python Fig_S4CFJ.py --config figureS4C_config.json
python Fig_S4CFJ.py --config figureS4F_config.json
python Fig_S4CFJ.py --config figureS4J_config.json

python Fig_S4GK.py --config figureS4G_config.json
python Fig_S4GK.py --config figureS4K_config.json
```

## Notes
- Outputs go to `save_dir` in each config, including `experiment.h5` and the SVG figure.

## Acknowledgements
- Portions of the refactor, project structuring, and documentation were assisted by OpenAI’s GPT models.
- Helpful discussions: Dan Gorbonos, Vit Piskovsky, Vivek Hari Sridhar, Iain Couzin.
- Guiding work: Sridhar, V. H., Li, L., Gorbonos, D., Nagy, M., Schell, B. R., Sorochkin, T., Gov, N. S., & Couzin, I. D. (2021). The geometry of decision-making in individuals and collectives. Proceedings of the National Academy of Sciences, 118(50). doi:10.1073/pnas.2102157118.
- Figures correspond to: V. Piskovsky, A. Paula, D. Gorbonos, P. K. Maini, I. D. Couzin, The neural basis of spatial exploration and decision-making, 2025 (unpublished).
