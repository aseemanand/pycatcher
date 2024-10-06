# __init__.py

# This makes it clear which test modules are included in the test package.
__all__ = [
    "test_catch.py", "test_diagnostics"
]

# Exposing specific anomaly_detection functions or classes.
from .test_catch import (
    test_find_outliers_iqr,
    test_anomaly_mad,
    test_get_residuals,
    test_sum_of_squares,
    test_get_ssacf,
    test_get_outliers_today,
    test_get_outliers_today_no_outliers,
)
