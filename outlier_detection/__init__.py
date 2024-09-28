"""
outlier_detection
------------------

This is an example package that demonstrates best practices for Python packaging.
It includes several modules and exposes some useful functions.

Modules:
    - anomaly_detection: Basic functions like greeting the world.
    - diagnostics: Simple mathematical operations.
"""

# Import functions from the individual modules so they can be accessed directly
from .anomaly_detection import *
from .diagnostics import *

# Defining a package-level version
__version__ = "0.1.0"

__all__ = ["find_outliers_iqr", "anomaly_mad", "get_residuals", "sum_of_squares", "get_ssacf", "get_outliers_today"]
