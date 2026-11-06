"""Fraud model training and evaluation for the course project.

Stages run in order and hand off through files, so each stage has one job:

1. fetch: download the raw IEEE-CIS tables.
2. prepare: validate, split by time, fit preprocessing on train only, and write
   the processed dataset, the fitted preprocessing pipeline, and a manifest.
3. train: read the processed dataset and manifest; never touch raw data.
4. evaluate: score the held-out time range and write the report.

Stages are added milestone by milestone; see docs/product/scope.md.
"""

__version__ = "0.1.0"
