## This repository contains the code for the thesis "The estimation of Value-at-Risk and Expected Shortfall based on deep generative models"
Author:  **Belonovskiy Peter Ilich, HSE DSBA** 

Supervisor: **Naumenko Vladimir Vladimirovich, HSE Associate Professor**

Thesis: https://www.hse.ru/en/edu/vkr/926006463?ysclid=m3e6gvwzru724833110

### Installation
___
The dependency management in this project relies on [poetry](https://python-poetry.org). Create or reuse a Python 3.10 environment (for example with `conda create -n var_es_dgm python=3.10`) and activate it **before** running any commands:

```bash
conda activate var_es_dgm
git clone https://github.com/BELONOVSKII/var_es_dgm.git
cd var_es_dgm
poetry install
```

Poetry must be available in the activated environment.

### Download data
___
Thesis uses daily stock prices data from yahoo finance. To parse the yahoo finance and download data run:
```python
python var_es_dgm/data_parcing/parse_yfinance.py 
```
This downloads individual stocks's data and produces combined file `data/complete_stocks.csv` that would be further used in the experiments.

### Models
___
* Variance Covariance: `var_es_dgm/basic_models/parametric.py`
* Historical Simulation: `var_es_dgm/basic_models/hist_sim.py`
* **TimeGrad**: `var_es_dgm/TimeGrad/`


### Experiments
___
All 16 experiments (2 dimensions × 4 methods × 2 VaR levels) can now be launched from the CLI. Activate the environment and call the runner:

```bash
python -m var_es_dgm.experiments.cli --dimension univariate --method timegrad --level 0.05 --device mps
```

Key flags:

- `--method`: `timegrad`, `timegrad_tuned`, `historical`, or `variance_covariance`. Use `--method all` to sweep every method for one dimension.
- `--run-all`: executes the entire 2×4×2 grid sequentially.
- `--level`: VaR level (`0.05` or `0.01`).
- `--n-repeats`: number of randomly sampled portfolios (default 5).
- `--portfolio-size`: number of tickers per portfolio (default 10).
- `--device`: torch device (`cpu`, `cuda`, `mps`, ...).

Outputs are written to `results/`:

- `results/checkpoints/<dimension>_<method>_<level>/repeat_<k>.pt` – TimeGrad weights.
- `results/logs/.../repeat_<k>.json` – run configuration, tickers, metrics, and loss curves.
- `results/results/.../summary.json` – mean statistics across repeats.

Legacy Jupyter notebooks remain under `experiments/` for reference, but the CLI is the source of truth for reproducible runs.

### Visualisations
___
All figures from the thesis could be created by running notebooks in `visualisations/`.
