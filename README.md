## PyCatcher
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/aseemanand/pycatcher/blob/main/LICENSE) [![Total Downloads](https://pepy.tech/badge/pycatcher)](https://pepy.tech/project/pycatcher) [![Monthly Downloads](https://pepy.tech/badge/pycatcher/month)](https://pepy.tech/project/pycatcher/month) [![Weekly Downloads](https://pepy.tech/badge/pycatcher/week)](https://pepy.tech/project/pycatcher/week) ![PYPI version](https://img.shields.io/pypi/v/pycatcher.svg) ![PYPI - Python Version](https://img.shields.io/pypi/pyversions/pycatcher.svg)

## Outlier Detection for Time-series Data
This package identifies outlier(s) for a given time-series dataset in simple steps. It supports day, week, month and quarter level time-series data.

#### DataFrame Arguments:
* First column in the dataframe must be a date column ('YYYY-MM-DD') and the last column a numeric column 
(sum or total count for the time period) to detect outliers using Seasonal Decomposition algorithms.
* Last column must be a numeric column to detect outliers using Moving Average and Z-score algorithm. 

#### Package Functions:
* detect_outliers(df): Detect outliers in a time-series dataframe using seasonal trend decomposition when there 
is at least 2 years of data, otherwise we can use Interquartile Range (IQR) for smaller timeframe.
* detect_outliers_today(df) Detect outliers for the current date in a time-series dataframe.
* detect_outliers_latest(df): Detect latest outliers in a time-series dataframe.
* detect_outliers_iqr(df): Detect outliers in a time-series dataframe when there's less than 2 years of data.
* detect_outliers_moving_average(df): Detect outliers using moving average method. 
* detect_outliers_stl(df): Detect outliers using Seasonal-Trend Decomposition using LOESS (STL).

#### Diagnostic Plots:
* build_seasonal_plot(df): Build seasonal plot (additive or multiplicative) for a given dataframe.
* build_iqr_plot(df): Build IQR plot for a given dataframe (for less than 2 years of data).
* build_monthwise_plot(df): Build month-wise plot for a given dataframe.
* build_decomposition_results(df): Get seasonal decomposition results for a given dataframe.
* conduct_stationarity_check(df): Conduct stationarity checks for a feature (dataframe's count column).
* build_moving_average_outliers_plot(df): Show outliers using Moving Average and Z-score algorithm.
* build_stl_outliers_plot(df): Show outliers using Seasonal-Trend Decomposition using LOESS (STL).

#### Highlights:
 Unlike many open-source packages for outlier detection, PyCatcher provides several distinctive features that set it apart:
* Automatic Model Selection: 
PyCatcher automatically detects whether to use an additive or multiplicative
decomposition model, ensuring the most accurate detection of outliers based on the characteristics of your data.
* Dynamic Method Selection Based on Data Size:
PyCatcher seamlessly switches between Seasonal Trend Decomposition (for datasets spanning at least two years) and
Interquartile Range (IQR) (for shorter time periods), offering flexibility without manual intervention.
* Wide Time Frequency Support:
Supports multiple time-series frequencies — including daily, weekly, monthly, and quarterly data—without requiring 
users to pre-process or adjust their datasets.
* Outlier Support: Support for outlier detection using both Classical Seasonal Trend Decomposition and Seasonal-Trend 
Decomposition using LOESS (STL) algorithms.
* Integrated Diagnostics: PyCatcher includes comprehensive diagnostic tools, enabling users to visualize outliers; trends 
and seasonal patterns; evaluate data stationarity, and analyze decomposition results.