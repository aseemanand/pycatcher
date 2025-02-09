### Summary of Diagnostic Functions
* `build_outliers_plot_classic(df):` Show outliers using Classical Seasonal Decomposition algorithm.
* `build_seasonal_plot_classic(df):` Build seasonal plot using Classical Seasonal Decomposition algorithm.
* `build_outliers_plot_stl(df):` Show outliers using Seasonal-Trend Decomposition using LOESS (STL) algorithm.
* `build_seasonal_plot_stl(df):` Build seasonal plots using STL for a given dataframe.
* `build_outliers_plot_mstl(df):` Show outliers using Multiple Seasonal-Trend Decomposition using LOESS (MSTL) algorithm.
* `build_outliers_plot_esd(df):` Build outliers using the Generalized ESD or Seasonal ESD algorithm.
* `build_seasonal_plot_mstl(df):` Build multiple seasonal plots using MSTL for a given dataframe.
* `build_outliers_plot_moving_average(df):` Show outliers using Moving Average algorithm.
* `build_iqr_plot(df):` Build IQR plot for a given dataframe (for less than 2 years of data).
* `build_monthwise_plot(df):` Build month-wise plot for a given dataframe.
* `build_decomposition_results(df):` Get seasonal decomposition results for a given dataframe.
* `conduct_stationarity_check(df):` Conduct stationarity checks for a feature (dataframe's count column).