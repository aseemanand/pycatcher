# __init__.py

# This makes it clear which test modules are included in the test package.
__all__ = [
    "test_catch",
    "test_diagnostics",
    "conftest",
    "test_webapp"
]

# Exposing specific anomaly_detection functions or classes.
from .test_catch import (
    test_find_outliers_iqr,
    test_anomaly_mad,
    test_get_residuals,
    test_sum_of_squares,
    test_get_ssacf,
    test_outliers_detected_today,
    test_no_outliers_today,
    test_outliers_latest_detected,
    test_no_outliers_detected,
    input_data_for_detect_outliers,
    input_data_decompose_and_detect,
    test_decompose_and_detect,
    input_data_detect_outliers_iqr,
    test_detect_outliers_iqr
)

from .test_diagnostics import (
    seasonal_decomposition_mock,
    test_plot_seasonal
)

from .conftest import (
    app,
    client,
    runner,
)

from .test_webapp import (
    sample_csv_file,
    invalid_file,
    test_index,
    test_upload_no_file,
    test_upload_invalid_file,
    test_upload_valid_csv,
    test_upload_and_analyze,
    test_file_saved_correctly
)