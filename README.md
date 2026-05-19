# Chronos-2 Forecasting

Companion code for practical Chronos-2 forecasting workflows, covering zero-shot inference now and fine-tuning later.

## What is here

- `01_chronos2_zero_shot_sir_demo.ipynb`: a runnable notebook that demonstrates zero-shot Chronos-2 forecasting on a synthetic multi-region SIR-style dataset.
- `requirements.txt`: Python dependencies for local notebook execution with CUDA-enabled PyTorch on Windows.

The zero-shot notebook walks through:

- univariate forecasting
- multivariate forecasting
- known-future covariates
- cross-learning for a short-history region

## Setup

Create and activate a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Register the notebook kernel:

```powershell
python -m ipykernel install --user --name chronos2-tds --display-name "Python (.venv Chronos-2 TDS)"
```

Run JupyterLab:

```powershell
jupyter lab
```

## Notes

Chronos-2 can run on CPU for small examples, but GPU inference is preferred for interactive use. This environment was tested with CUDA-enabled PyTorch on an NVIDIA RTX 2000 Ada laptop GPU.
