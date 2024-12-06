## PyCatcher
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/aseemanand/pycatcher/blob/main/LICENSE) [![Total Downloads](https://pepy.tech/badge/pycatcher)](https://pepy.tech/project/pycatcher) [![Monthly Downloads](https://pepy.tech/badge/pycatcher/month)](https://pepy.tech/project/pycatcher/month) [![Weekly Downloads](https://pepy.tech/badge/pycatcher/week)](https://pepy.tech/project/pycatcher/week) ![PYPI version](https://img.shields.io/pypi/v/pycatcher.svg) ![PYPI - Python Version](https://img.shields.io/pypi/pyversions/pycatcher.svg)

## Outlier Detection for Time-series Data
This package identifies outlier(s) for a given time-series dataset in simple steps. It supports day, week, month and 
quarter level time-series data.

### Installation

```bash
pip install pycatcher
```

### DataFrame Arguments
* First column in the dataframe must be a date column ('YYYY-MM-DD') and the last column a numeric column 
(sum or total count for the time period) to detect outliers using Seasonal Decomposition algorithms.
* Last column must be a numeric column to detect outliers using Moving Average and Z-score algorithm. 

<hr style="border:1.25px solid gray">

## Summary of features 
PyCatcher provides an efficient solution for detecting anomalies in time-series data using various statistical methods.
Below are the available techniques for anomaly detection, each optimized for different data characteristics.

### **1. Seasonal-Decomposition Based Anomaly Detection**

#### **Detect Outliers Using Classical Seasonal Decomposition**
For datasets with at least two years of data, PyCatcher automatically determines whether the data follows an additive or multiplicative model to detect anomalies.

- **Method**: `detect_outliers(df)`
- **Output**: DataFrame of detected anomalies or a message indicating no anomalies.

#### **Detect Today's Outliers**
Quickly identify if there are any anomalies specifically for the current date.

- **Method**: `detect_outliers_today(df)`
- **Output**: Anomaly details for today or a message indicating no outliers.

#### **Detect the Latest Anomalies**
Retrieve the most recent anomalies identified in your time-series data.

- **Method**: `detect_outliers_latest(df)`
- **Output**: Details of the latest detected anomalies.

#### **Visualize Outliers with Seasonal Decomposition**
Show outliers in your data through classical seasonal decomposition.

- **Method**: `build_classical_seasonal_outliers_plot(df)`
- **Output**: Outlier plot generated using classical seasonal decomposition.

#### **Visualize Seasonal Patterns**
Understand seasonality in your data by visualizing trends through decomposition.

- **Method**: `build_seasonal_plot(df)`
- **Output**: Seasonal plots displaying additive or multiplicative trends.

#### **Visualize Monthly Patterns**
Show month-wise box plot 

- **Method**: `build_monthwise_plot(df)`
- **Output**: Month-wise box plots showing spread and skewness of data.


#### **Detect Outliers Using Seasonal-Trend Decomposition using LOESS (STL)**
Use the Seasonal-Trend Decomposition method (STL) to detect anomalies.

- **Method**: `detect_outliers_stl(df)`
- **Output**: Rows flagged as outliers using STL.

#### **Visualize STL Outliers**
Show outliers using the Seasonal-Trend Decomposition using LOESS (STL).

- **Method**: `build_stl_outliers_plot(df)`
- **Output**: Outlier plot generated using STL.

***

### **2. Detect Outliers Using Moving Average**
Detect anomalies in time-series data using the Moving Average method.

- **Method**: `detect_outliers_moving_average(df)`
- **Output**: Rows flagged as outliers using Moving Average and Z-score algorithm.

#### **Visualize Moving Average Outliers**
Show outliers using the Moving Average and Z-score algorithm.

- **Method**: `build_moving_average_outliers_plot(df)`
- **Output**: Outlier plot generated using Moving Average method.
  
---

### **3. IQR-Based Anomaly Detection**

#### **Detect Outliers Using Interquartile Range (IQR)**
For datasets spanning less than two years, the IQR method is employed.

- **Method**: `find_outliers_iqr(df)`
- **Output**: Rows flagged as outliers based on IQR.

#### **Visualize IQR Plot**
Build an IQR plot for a given dataframe (for less than 2 years of data).

- **Method**: `build_iqr_plot(df)`
- **Output**: IQR plot for the time-series data.

<hr style="border:1.25px solid gray">

### Summary of Package Functions
* `detect_outliers(df):` Detect outliers in a time-series dataframe using seasonal trend decomposition when there 
is at least 2 years of data, otherwise we can use Inter Quartile Range (IQR) for smaller timeframe.
* `detect_outliers_today(df):` Detect outliers for the current date in a time-series dataframe.
* `detect_outliers_latest(df):` Detect latest outliers in a time-series dataframe.
* `detect_outliers_iqr(df):` Detect outliers in a time-series dataframe when there's less than 2 years of data.
* `detect_outliers_moving_average(df):` Detect outliers using moving average method. 
* `detect_outliers_stl(df):` Detect outliers using Seasonal-Trend Decomposition using LOESS (STL).

<hr style="border:1.25px solid gray">

### Summary of Diagnostic Plots
* `build_seasonal_plot(df):` Build seasonal plot (additive or multiplicative) for a given dataframe.
* `build_iqr_plot(df):` Build IQR plot for a given dataframe (for less than 2 years of data).
* `build_monthwise_plot(df):` Build month-wise plot for a given dataframe.
* `build_decomposition_results(df):` Get seasonal decomposition results for a given dataframe.
* `build_classical_seasonal_outliers_plot(df):` Show outliers using Classical Seasonal Decomposition algorithm.
* `build_moving_average_outliers_plot(df):` Show outliers using Moving Average and Z-score algorithm.
* `build_stl_outliers_plot(df):` Show outliers using Seasonal-Trend Decomposition using LOESS (STL).
* `conduct_stationarity_check(df):` Conduct stationarity checks for a feature (dataframe's count column).

<hr style="border:1.25px solid gray">

### Highlights
 Unlike many open-source packages for outlier detection, PyCatcher provides several distinctive features:
* **Automatic Model Selection:** 
PyCatcher automatically detects whether to use an additive or multiplicative
decomposition model, ensuring the most accurate detection of outliers based on the characteristics of your data.
* **Dynamic Method Selection Based on Data Size:**
PyCatcher seamlessly switches between Seasonal Trend Decomposition (for datasets spanning at least two years) and
Inter Quartile Range (IQR) for shorter time periods, offering flexibility without manual intervention.
* **Wide Time Frequency Support:**
Supports multiple time-series frequencies — including daily, weekly, monthly, and quarterly data—without requiring 
users to pre-process or adjust their datasets.
* **Choice for Different Seasonal Trend Algorithms:** Support for outlier detection using both Classical Seasonal Trend 
Decomposition 
and Seasonal-Trend Decomposition using LOESS (STL) algorithms.
* **Integrated Diagnostics:** PyCatcher includes comprehensive diagnostic tools, enabling users to visualize outliers, 
trends 
and seasonal patterns, evaluate data stationarity, and analyze decomposition results.
* **User Interface:** Availability of a simple user interface for the users to upload file for outlier detection using IQR.
Future versions will include advanced algorithms.

<hr style="border:1.25px solid gray">

### Example Usage

To see an example of how to use the `pycatcher` package for outlier detection in time-series data, check out the [Example Notebook](https://github.com/aseemanand/pycatcher/blob/main/notebooks/Example%20Notebook.ipynb).

The notebook provides step-by-step guidance and demonstrates the key features of the library.


