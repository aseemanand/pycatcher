"""
pycatcher
------------------

This package identifies the anomalies for a given dataset.

Modules:
    - anomaly_detection: Functions to find the Anomalies within a given dataset.
    - diagnostics: Functions to run some diagnostics on the data.
"""

# Import functions from the individual modules so they can be accessed directly
from pycatcher.catch import (find_outliers_iqr, anomaly_mad, sum_of_squares, detect_outliers_today,
                             detect_outliers_latest, detect_outliers, decompose_and_detect, detect_outliers_iqr,detect_outliers_moving_average,calculate_optimal_window_size,calculate_rmse)
from pycatcher.diagnostics import get_residuals, get_ssacf, plot_seasonal, build_iqr_plot, build_seasonal_plot, build_monthwise_plot,conduct_stationarity_check,build_decomposition_results

__all__ = ["find_outliers_iqr", "anomaly_mad", "get_residuals", "sum_of_squares", "get_ssacf", "detect_outliers_today",
           "detect_outliers_latest", "detect_outliers", "build_seasonal_plot","conduct_stationarity_check","build_decomposition_results","plot_seasonal", "build_monthwise_plot", "decompose_and_detect",
           "detect_outliers_iqr","detect_outliers_moving_average","calculate_optimal_window_size","calculate_rmse"]
