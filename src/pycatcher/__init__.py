"""
pycatcher
------------------

This is a package that identifies the anomalies from the given input dataset.

Modules:
    - anomaly_detection: Functions to find the Anomalies within a given dataset.
    - diagnostics: Functions to run some diagnostics on the data.
"""

# Import functions from the individual modules so they can be accessed directly
from .catch import *
from .diagnostics import *

__all__ = ["find_outliers_iqr", "anomaly_mad", "get_residuals", "sum_of_squares", "get_ssacf", "detect_outliers_today",
           "detect_outliers_latest", "detect_outliers", "plot_seasonal", "build_plot", "build_monthwise_plot"]
