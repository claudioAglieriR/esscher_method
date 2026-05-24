# Esscher Method : Structural Lévy Model Calibration

This repository implements the calibration workflow described in the research article:

**“An Esscher-Based Algorithm For Computing Default Probabilities In Structural Lévy Models”**  
Jean‑Philippe Aguilar, Justin Lars Kirkby, and me - *Journal of Fixed Income*.

The code provides a **library-style implementation** of an iterative structural credit-risk calibration procedure based on **Lévy asset dynamics**.  
It **infers the (unobserved) firm asset value process** from equity observations and computes **distance-to-default** and **default probabilities** under the physical measure.

---

## What’s inside the repository

### `esscher_method/`
A minimal, self-contained Python package:

- **`esscher_method/model/`**
  - `Model` abstract base class
  - `Merton`, `VarianceGamma`, `BilateralGamma` Lévy models
  - parameter policies + numerical settings

- **`esscher_method/calibrator/`**
  - `Calibrator`: orchestrates the full algorithm
  - `MomentMatcher`: sample cumulants + residuals for moment matching
  - `ParameterOptimizer`: bounded least-squares + optional differential evolution fallback
  - `EsscherSolver`: bounded solver for `p*` (Esscher parameter)
  - `LewisEuropeanTargetPricer`: Lewis Fourier call pricer (used under risk-neutral measure)
  - `AssetInferenceEngine`: equity→asset inversion (serial or parallel)

### Demo notebook
- **`demo_esscher_method.ipynb`**  
  Loads the included CSV datasets, calibrates all models across all tickers, and outputs model-specific tables.

---

## Installation

### 1) Create a virtual environment

**Windows (Command Prompt)**
```bash
py -m venv venv
venv\Scripts\activate
```


### 2) Install the package and dependencies

From the repository root:

```bash
py -m pip install -U pip
py -m pip install -e ".[notebook,test,dev]"
```



> Notes  
> - `pyproject.toml` defines all runtime and optional dependencies.  
> - The repository is developed/tested for **Python ≥ 3.10**.

---

## Quick start

### Run the demo notebook
```bash
jupyter notebook demo_esscher_method.ipynb
```

The notebook demonstrates:
1. loading equity/debt data from CSV  
2. calibrating **Merton**, **VarianceGamma**, **BilateralGamma**  
3. solving Esscher `p*`  
4. inverting equity to assets using Lewis Fourier pricing  
5. iterating until convergence  
6. computing distance-to-default and default probability

---

## Calibration workflow (high-level)

The calibration loop implemented in `Calibrator` follows:

1. **Physical calibration on equity returns**  
   Estimate physical parameters by moment matching (sample cumulants vs theoretical cumulants).

2. **Esscher transform (solve for p\*)**  
   Find `p*` satisfying the Esscher equation using the model CGF.

3. **Asset inference (equity → asset)**  
   Treat equity as a call option on assets with strike = debt.  
   Invert the Lewis call price to obtain implied asset values.

4. **Physical re-calibration on inferred asset returns**  
   Re-estimate physical parameters using inferred asset log-returns.

5. **Repeat until convergence**  
   Stop when physical parameters stabilize within the configured tolerance or `max_iterations` is reached.

6. **Default probability**  
   Compute distance-to-default and evaluate the physical CDF via Gil–Peláez inversion.

---

## Configuration knobs

The main configuration lives in:

- `CalibrationConfig` (moment matching, optimizers, parallel asset inference)
- `AssetInversionConfig` (root bracketing, fallback minimization)
- `LewisPricerConfig` (Fourier integration settings)
- model-specific `Policy` objects (parameter bounds and defaults)

Parallel asset inference is enabled by default (`use_parallel=True`).

---

## Reproducibility

The package reproduces the published paper out of the box. A fresh clone
of the repository, followed by the demo notebook with no modifications,
yields exactly the numbers reported in Tables 1, 3 and 4 of the paper.
This is the default behaviour and is preserved by the regression test
suite.

The commit tagged `paper-v1` is the formal audit anchor for this
guarantee, useful if you want to verify reproducibility against a
specific point in the repository's history:

```bash
git checkout paper-v1
```

To return to the latest development state:

```bash
git checkout master
```

Where the package ships alternative methodologies (for example, a
different moment-matching identification for the Variance Gamma model),
those are exposed as opt-in configuration flags, disabled by default.
Activating such a flag explicitly produces values that may differ from
the paper, and is the user's deliberate choice.

---

## Tests

The test suite is organised in three layers, identified by pytest markers:

| Layer | Marker | Default `pytest` | Command | Typical runtime |
|---|---|---|---|---|
| Unit + smoke + regression baselines | _(none)_ | runs | `pytest` | ~3 minutes |
| Extended smoke (e.g. PD-bootstrap coverage on Merton) | `slow` | skipped | `pytest -m slow` | ~5 minutes |
| Comprehensive statistical validation | `nightly` | skipped | `pytest -m nightly` | hours |

The first layer is the standard CI check; it pins numerical regression
baselines (PD% and fitted parameters per ticker x model) at
`rtol = 1e-4, atol = 1e-4`. The `slow` layer adds extended smoke tests
that take a few minutes. The `nightly` layer runs comprehensive
statistical validation (parameter-recovery studies at large sample
sizes, bootstrap coverage probability across all three Levy models) and
is intended for release validation; runtime is in the order of hours.

To run two layers together, combine markers with `or`:

```bash
pytest -m "slow or nightly"      # both opt-in layers
pytest -m "not nightly"          # default + slow
```


