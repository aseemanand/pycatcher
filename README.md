## PyCatcher
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/aseemanand/pycatcher/blob/main/LICENSE) [![Total Downloads](https://pepy.tech/badge/pycatcher)](https://pepy.tech/project/pycatcher) [![Monthly Downloads](https://pepy.tech/badge/pycatcher/month)](https://pepy.tech/project/pycatcher/month) [![Weekly Downloads](https://pepy.tech/badge/pycatcher/week)](https://pepy.tech/project/pycatcher/week) ![PYPI version](https://img.shields.io/pypi/v/pycatcher.svg) ![PYPI - Python Version](https://img.shields.io/pypi/pyversions/pycatcher.svg)

## Outlier Detection for Time-series Data
This package identifies outlier(s) for a given time-series dataset in simple steps. It supports day, week, month and quarter level time-series data.

#### DataFrame Arguments:
First column in the dataframe must be a date column ('YYYY-MM-DD') and the last column a count column.
#### Package Functions:
* detect_outliers(df): Detect outliers in a time-series dataframe using seasonal trend decomposition when there is at least 2 years of data, otherwise we can use Interquartile Range (IQR) for smaller timeframe.
* detect_outliers_today(df) Detect outliers for the current date in a time-series dataframe.
* detect_outliers_latest(df): Detect latest outliers in a time-series dataframe.
* find_outliers_iqr(df): Detect outliers in a time-series dataframe when there's less than 2 years of data.

#### Diagnostic Plots:
* build_seasonal_plot(df): Build seasonal plot (additive or multiplicative) for a given dataframe.
* build_iqr_plot(df): Build IQR plot for a given dataframe (for less than 2 years of data).
* build_monthwise_plot(df): Build month-wise plot for a given dataframe.
* build_decomposition_results(df): Get seasonal decomposition results for a given dataframe.
* conduct_stationarity_check(df): Conduct stationarity check (trend only) for a feature (dataframe's count column).
