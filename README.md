## Outlier Detection for Time-series Data
This package identifies outlier(s) for a given day-level time-series dataset in simple steps.

#### DataFrame Arguments:
First column in the dataframe must be a date column ('YYYY-MM-DD') and the last column a count column.
#### Package Functions:
* detect_outliers(df): Detect outliers in a time-series dataframe using seasonal trend decomposition when there is at least 2 years of data, otherwise we can use Interquartile Range (IQR) for smaller timeframe.
* detect_ouliers_today(df) Detect outliers for the current date in a time-series dataframe.
* detect_outliers_latest(df): Detect latest outliers in a time-series dataframe.
* find_outliers_iqr(df): Detect outliers in a time-series dataframe when there's less than 2 years of data.

#### Diagnostic Plots:
* built_seasonal_plot(df): Build seasonal plot (additive or multiplicative) for a given dataframe.
* built_iqr_plot(df): Build IQR plot for a given dataframe (less than 2 years of data).
* build_monthwise_plot(df): Build month-wise plot for a given dataframe.
* build_decomposition_results(df): Get seasonal decomposition results for a given dataframe.
* conduct_stationarity_check(df): Conduct stationarity check (trend only) for a feature (dataframe's count column).







