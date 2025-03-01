### Summary of Package Functions
* `detect_outliers_classic(df):` Detect outliers in a time-series dataframe using Classical Seasonal Decomposition. 
* `detect_outliers_today_classic(df):` Detect outliers for the current date using Classical Seasonal Decomposition.
* `detect_outliers_latest_classic(df):` Detect latest outliers using Classical Seasonal Decomposition.
* `detect_outliers_stl(df):` Detect outliers using Seasonal-Trend Decomposition using LOESS (STL).
* `detect_outliers_today_stl(df):` Detect outliers for the current date using STL.
* `detect_outliers_latest_stl(df):` Detect latest outliers using STL.
* `detect_outliers_mstl(df):` Detect outliers using Multiple Seasonal-Trend Decomposition using LOESS (MSTL).
* `detect_outliers_today_mstl(df):` Detect outliers for the current date using MSTL.
* `detect_outliers_latest_mstl(df):` Detect latest outliers using MSTL.
* `detect_outliers_esd(df):` Detect outliers using the Generalized ESD or Seasonal ESD algorithm.
* `detect_outliers_today_esd(df):` Detect outliers for the current date using ESD algorithm.
* `detect_outliers_latest_esd(df):` Detect latest outliers using ESD algorithm.
* `detect_outliers_iqr(df):` Detect outliers in a time-series dataframe when there's less than 2 years of data.
* `detect_outliers_moving_average(df):` Detect outliers using Moving Average method. 
