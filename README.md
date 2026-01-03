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
venv\Scripts\activate.bat
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


