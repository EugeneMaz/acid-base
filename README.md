# Acid-Base Equilibrium

A SymPy-based model for calculating the equilibrium composition of a polyprotic buffer system while accounting for ionic strength and activity coefficients.

The project contains the reusable `AcidBaseEquilibrium` class in `acid_base_equilibrium.py` and example calculations in `buffer-equilibrium.ipynb`.

## Features

- Supports one or more acid dissociation constants (`pKa` values).
- Includes water autoionization optionally.
- Supports Davies and Truesdell-Jones activity-coefficient models for H+ and OH-.
- Solves one set of conditions or sweeps concentration, pH, and added ionic strength.
- Can parallelize range calculations with `joblib`.
- Displays equilibrium equations and concentration results in a Jupyter notebook.
- Saves range results as semicolon-delimited CSV files.

## Requirements

- Conda or Miniconda
- Python 3
- Jupyter Notebook or JupyterLab

The supplied `environment.yml` provides the numerical dependencies. The module also uses IPython and joblib, so install those packages if they are not already available in your environment.

## Running the examples

Start Jupyter from this directory:

```bash
jupyter lab
```

Open `buffer-equilibrium.ipynb` and run the cells from top to bottom. The first cell imports the class from `acid_base_equilibrium.py`; the remaining cells define a buffer, solve individual conditions, sweep parameter ranges, and save CSV files.


## Notes

- Run the notebook with the project directory as the working directory so that the local module import succeeds.
- Range calculations can be computationally expensive, especially with many pH, concentration, and ionic-strength values.
- The model is intended for exploratory calculations. Validate assumptions, activity models, and input ranges for any quantitative application.
