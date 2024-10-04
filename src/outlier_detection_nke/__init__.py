"""
outlier_detection_nke
------------------

This is a package that identifies the anomalies from the given input dataset.

Modules:
    - anomaly_detection: Functions to find the Anomalies within a given dataset.
    - diagnostics: Functions to run some diagnostics on the data.
"""

# Import functions from the individual modules so they can be accessed directly
from .anomaly_detection import *
from .diagnostics import *

# Defining a package-level version
__version__ = "0.1.0"

__all__ = ["find_outliers_iqr", "anomaly_mad", "get_residuals", "sum_of_squares", "get_ssacf", "get_outliers_today"]
