# Data sources

## AI4I 2020 Predictive Maintenance Dataset

Canonical source: UCI Machine Learning Repository, dataset 601.

- DOI: `10.24432/C5HS5C`
- License: CC BY 4.0
- Canonical CSV: `https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv`

Run:

```bash
python -m src.data.download_ai4i
```

The raw CSV is intentionally gitignored.

### Important limitation

AI4I is synthetic but designed to reflect industrial predictive-maintenance data. It contains independent product rows rather than repeated timestamped histories for individual machines. We therefore use it as a **tabular failure-classification benchmark**, not as evidence that the model predicts failure 24 hours into the future.

### Leakage policy

`TWF`, `HDF`, `PWF`, `OSF`, and `RNF` describe failure modes that directly determine the `Machine failure` target. They are retained only for dataset auditing and are **never supplied to a model as predictors**.
