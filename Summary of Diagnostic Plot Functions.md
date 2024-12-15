### Summary of Diagnostic Plots
* `build_seasonal_outliers_plot_classic(df):` Show outliers using Classical Seasonal Decomposition algorithm.
* `build_seasonal_plot_classic(df):` Build seasonal plot using classic seasonal trend decomposition.
* `build_outliers_plot_stl(df):` Show outliers using Seasonal-Trend Decomposition using LOESS (STL).
* `build_seasonal_plot_stl(df):` Build seasonal plots using STL for a given dataframe.
* `build_outliers_plot_mstl(df):` Show outliers using Multiple Seasonal-Trend Decomposition using LOESS (MSTL).
* `build_seasonal_plot_mstl(df):` Build multiple seasonal plots using MSTL for a given dataframe.
* `build_moving_average_outliers_plot(df):` Show outliers using Moving Average and Z-score algorithm.
* `build_iqr_plot(df):` Build IQR plot for a given dataframe (for less than 2 years of data).
* `build_monthwise_plot(df):` Build month-wise plot for a given dataframe.
* `build_decomposition_results(df):` Get seasonal decomposition results for a given dataframe.
* `conduct_stationarity_check(df):` Conduct stationarity checks for a feature (dataframe's count column).
