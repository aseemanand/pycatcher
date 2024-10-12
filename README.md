## Outlier Detection for time-series data
This package identifies the day-level time-series outliers for a given dataset. 
#### Arguments:
First column must be a date column ('YYYY-MM-DD') and second or the last column a count column.
#### Functions:
##### detect_outliers(df): Detect outliers in a time-series dataframe using seasonal trend decomposition when there is at least 2 years of data, otherwise we can use Interquartile Range (IQR) for smaller timeframe.
##### detect_ouliers_today(df) Detect outliers for the current date in a time-series dataframe.
##### detect_outliers_latest(df): Detect latest outliers in a time-series dataframe.
##### find_outliers_iqr: Detect outliers in a time-series dataframe when there's less than 2 years of data.





