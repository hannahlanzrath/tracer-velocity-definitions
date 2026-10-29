# What Does Tracer Velocity Measure?

Code to reproduce the results of

> H. Lanzrath, G. Huber, E. von Lieres. *What Does Tracer Velocity Measure? A Theoretical Analysis of Velocity Definitions in Plant Transport.* Submitted to Mathematical Biosciences.

The simulations use the Multichannel Transport (MCT) model implemented in [CADET](https://github.com/cadet/CADET-Core) and set up with [CADET-Process](https://github.com/fau-advanced-separations/CADET-Process).

## Installation

With conda (installs CADET-Core from conda-forge):

```bash
conda env create -f environment.yaml
conda activate tracer-velocity
```

With pip, CADET-Core has to be installed separately (see the [CADET documentation](https://cadet.github.io)):

```bash
pip install -r requirements.txt
```

## Usage

| Script | Reproduces |
|---|---|
| `velocity_estimators.py` | Table 1 and Figures 5–7: simulations of Model M13 and velocity estimates of the numerical temporal centroid, half-integral, half-maximum, and intercept estimators, together with the analytical mean tracer velocity |
| `vmean_isolines.py` | Figure 4: iso-lines of the analytical mean tracer velocity over the exchange-rate ratios R and S |
| `finite_domain_validation.py` | Appendix C: comparison of the dispersion-corrected expression with the finite-domain solution of the moment equations |

```bash
python velocity_estimators.py
python vmean_isolines.py
python finite_domain_validation.py
```

Results are written to `results/` (change with `--output-dir`). `velocity_estimators.py` also writes the values of Table 1 to `results/table1_velocities_mm_per_min.csv`.

Figure 4 in the paper was created with an earlier version of the plotting code. `vmean_isolines.py` plots the same function, but the iso-levels and the colors of the guide lines differ slightly.

## License

This project is licensed under the GNU Affero General Public License v3.0, see [LICENSE](LICENSE).
