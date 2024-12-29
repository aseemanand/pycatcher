"""
pycatcher
------------------

A Python package for anomaly detection in datasets.

This package provides tools and utilities for identifying anomalies and outliers
in various types of datasets, with a focus on time series data.

Modules:
    - catch: Core anomaly detection algorithms and utilities
    - diagnostics: Statistical analysis and diagnostic tools
    - webapp: Flask web application for interactive analysis

For more information, visit: https://github.com/aseemanand/pycatcher/blob/main/README.md
"""

# Import packages and modules for web app
import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from flask import Flask

# Import PyCatcher API functions
from pycatcher.catch import (
    find_outliers_iqr,
    anomaly_mad,
    anomaly_zscore,
    sum_of_squares,
    detect_outliers_today_classic,
    detect_outliers_latest_classic,
    detect_outliers_classic,
    decompose_and_detect,
    detect_outliers_iqr,
    detect_outliers_moving_average,
    calculate_optimal_window_size,
    calculate_rmse,
    check_normal_distribution_monte_carlo,
    generate_outliers_stl,
    generate_outliers_mstl,
    generate_outliers_generalized_esd,
    generate_outliers_seasonal_esd,
    detect_outliers_stl,
    detect_outliers_mstl,
    detect_outliers_today_stl,
    detect_outliers_latest_stl,
    detect_outliers_today_mstl,
    detect_outliers_latest_mstl,
    detect_outliers_today_esd,
    detect_outliers_latest_esd,
    detect_outliers_esd,
    detect_ts_frequency,


)
from pycatcher.diagnostics import (
    get_residuals,
    get_ssacf,
    plot_seasonal,
    build_iqr_plot,
    build_seasonal_plot_classic,
    build_monthwise_plot,
    conduct_stationarity_check,
    build_decomposition_results,
    build_outliers_plot_moving_average,
    build_outliers_plot_classic,
    build_outliers_plot_stl,
    build_outliers_plot_mstl,
    build_seasonal_plot_mstl,
    build_seasonal_plot_stl,
    build_outliers_plot_esd,

)

# Public API definition
__all__ = [
    # Anomaly detection functions
    "find_outliers_iqr",
    "anomaly_mad",
    "anomaly_zscore",
    "detect_outliers_today_classic",
    "detect_outliers_latest_classic",
    "detect_outliers_classic",
    "detect_outliers_iqr",
    "detect_outliers_moving_average",
    "detect_outliers_today_stl",
    "detect_outliers_latest_stl",
    "detect_outliers_today_mstl",
    "detect_outliers_latest_mstl",
    "detect_outliers_esd",
    "detect_outliers_today_esd",
    "detect_outliers_latest_esd",
    "detect_ts_frequency",

    # Statistical functions
    "sum_of_squares",
    "calculate_optimal_window_size",
    "calculate_rmse",
    "decompose_and_detect",
    "check_normal_distribution_monte_carlo",

    # Diagnostic functions
    "get_residuals",
    "get_ssacf",
    "plot_seasonal",
    "build_iqr_plot",
    "build_seasonal_plot_classic",
    "build_monthwise_plot",
    "conduct_stationarity_check",
    "build_decomposition_results",
    "build_outliers_plot_moving_average",
    "generate_outliers_stl",
    "detect_outliers_stl",
    "detect_outliers_mstl",
    "generate_outliers_mstl",
    "generate_outliers_seasonal_esd",
    "generate_outliers_generalized_esd",
    "build_outliers_plot_classic",
    "build_outliers_plot_stl",
    "build_outliers_plot_mstl",
    "build_seasonal_plot_stl",
    "build_seasonal_plot_mstl",
    "build_outliers_plot_esd",

    # Web app
    "create_app"
]


def setup_logging(app: Flask) -> None:
    """
    Configure logging for the application.

    Args:
        app: Flask application instance
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create handlers
    file_handler = RotatingFileHandler(
        log_dir / 'pycatcher.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )

    # Set log formats
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Set log levels
    if app.debug:
        file_handler.setLevel(logging.DEBUG)
    else:
        file_handler.setLevel(logging.INFO)

    # Add handlers
    app.logger.addHandler(file_handler)


# Web app initialization
def create_app(test_config=None):
    """
    Create and configure the Flask application using factory pattern.

    Args:
        test_config (dict, optional): Test configuration to override defaults

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    if test_config is None:
        # Default configuration
        app.config.from_mapping(
            SECRET_KEY=os.getenv('SECRET_KEY', 'default_secret_key'),
            UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', './uploads'),
            ALLOWED_EXTENSIONS={'csv'},
            MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
        )
    else:
        # Override with test config if passed in
        app.config.update(test_config)

    # Ensure upload folder exists
    upload_path = Path(app.config['UPLOAD_FOLDER'])
    upload_path.mkdir(parents=True, exist_ok=True)

    return app