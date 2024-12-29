import logging
import re as regex
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import (adfuller, kpss)
from statsmodels.tsa.seasonal import STL, MSTL
import statsmodels.api as sm
import numpy as np
from scipy import stats
from scipy.stats import shapiro

from .catch import (get_residuals,
                    get_ssacf,
                    anomaly_mad,
                    anomaly_zscore,
                    calculate_optimal_window_size,
                    generate_outliers_stl,
                    generate_outliers_mstl,
                    generate_outliers_generalized_esd,
                    generate_outliers_seasonal_esd
                    )

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_seasonal(res, axes, title):
    """
    Args:
        res: Model type result
        axes: An Axes typically has a pair of Axis Artists that define the data coordinate system,
              and include methods to add annotations like x- and y-labels, titles, and legends.
        title: Title of the plot

    """

    logger.info("Plotting seasonal decomposition with title: %s", title)

    # Plotting Seasonal time series models
    axes[0].title.set_text(title)
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')

    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')

    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')

    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')


def build_seasonal_plot_classic(df):
    """
    Build seasonal plot for a given dataframe using classic seasonal decomposition
        Args:
             df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                               and the second/last column should be the feature (count).
    """

    logger.info("Building time-series plot for seasonal decomposition.")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    df_season = df_pandas.copy()
    # Ensure the first column is in datetime format and set it as index
    df_season.iloc[:, 0] = pd.to_datetime(df_season.iloc[:, 0])
    df_season = df_season.set_index(df_season.columns[0]).asfreq('D').dropna()

    # Find length of time period to decide right outlier algorithm
    length_year = len(df_season.index) // 365.25

    logger.info("Time-series data length: %.2f years", length_year)

    if length_year >= 2.0:

        # Building Additive and Multiplicative time series models
        # In a multiplicative time series, the components multiply together to make the time series.
        # If there is an increasing trend, the amplitude of seasonal activity increases.
        # Everything becomes more exaggerated. This is common for web traffic.

        # In an additive time series, the components add together to make the time series.
        # If there is an increasing trend, we still see roughly the same size peaks and troughs
        # throughout the time series. This is often seen in indexed time series where the
        # absolute value is growing but changes stay relative.

        decomposition_add = sm.tsa.seasonal_decompose(df_season.iloc[:, -1],
                                                      model='additive', extrapolate_trend='freq')
        residuals_add = get_residuals(decomposition_add)

        decomposition_mul = sm.tsa.seasonal_decompose(df_season.iloc[:, -1],
                                                      model='multiplicative', extrapolate_trend='freq')
        residuals_mul = get_residuals(decomposition_mul)

        # Get ACF values for both Additive and Multiplicative models

        # Calculate Sum of Squares of the ACF for both models
        ssacf_add: float = get_ssacf(residuals_add, type='Additive')
        ssacf_mul: float = get_ssacf(residuals_mul, type='Multiplicative')

        if ssacf_add < ssacf_mul:
            logger.info("Using Additive model for seasonal decomposition.")
            _, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_add, axes, title="Additive")
        else:
            logger.info("Using Multiplicative model for seasonal decomposition.")
            _, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_mul, axes, title="Multiplicative")
    else:
        logger.info("Use boxplot since the data is less than 2 years.")
        print('Use build_iqr_plot method to see the boxplot with outliers')


def build_iqr_plot(df):
    """
    Build IQR plot for a given dataframe.

    Args:
        df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                                   and the second/last column should be the feature (count).

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    logger.info("Building IQR plot to see outliers")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Set the background color of the figure to white
    fig.patch.set_facecolor('white')

    # Ensure the last column is numeric
    df.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])

    # Create a horizontal boxplot using Seaborn
    sns.boxplot(x=df.iloc[:, -1], ax=ax, showmeans=True)
    ax.set_title("Outlier Detection Plot")
    ax.set_xlabel("Values")
    ax.set_ylabel("")

    # Adjust layout to avoid clipping of labels
    plt.tight_layout()

    plt.close(fig)
    return fig


def build_monthwise_plot(df):
    """
        Build month-wise plot for a given dataframe
            Args:
                 df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                                   and the last column should be the feature (count).
    """

    logger.info("Building month-wise box plot.")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    df_month = df_pandas.copy()
    df_month['Month-Year'] = pd.to_datetime(df_month.iloc[:, 0]).dt.to_period('M')
    df_month['Count'] = pd.to_numeric(df_month.iloc[:, 1])
    plt.figure(figsize=(30, 4))
    sns.boxplot(x='Month-Year', y='Count', data=df_month).set_title("Month-wise Box Plot")
    plt.show()


def conduct_stationarity_check(df):

    """
    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        ADF and KPSS statistics. Time series are stationary if they
        do not have trend or seasonal effects.
        Summary statistics calculated on the time series are consistent over time,
        like the mean or the variance of the observations.
    """
    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the last column is numeric
    df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])

    logger.info("Starting ADF stationarity check")

    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(df_pandas.iloc[:, -1])

    logger.info("ADF Statistic: %f", adf_result[0])
    logger.info('p-value: %f', adf_result[1])
    logger.info("Critical Values:")
    for key, value in adf_result[4].items():
        logger.info('\t%s: %.3f', key, value)

    if (adf_result[1] <= 0.05) & (adf_result[4]['5%'] > adf_result[0]):
        logger.info("Completed ADF stationarity check")
        print("\u001b[32mADF - The series is Stationary\u001b[0m")
    else:
        logger.info("Completed ADF stationarity check")
        print("\x1b[31mADF - The series is not Stationary\x1b[0m")

    print("\n")

    # Perform KPSS test
    logger.info("Starting KPSS stationarity check")
    statistic, p_value, n_lags, critical_values = kpss(df_pandas.iloc[:, -1])

    logger.info('KPSS Statistic: %f', statistic)
    logger.info('p-value: %f', p_value)
    logger.info('n_lags: %f', n_lags)
    logger.info('Critical Values:')

    for key, value in critical_values.items():
        logger.info(' %s : %s', key, value)

    logger.info("Completed KPSS stationarity check")
    print(f'\u001b[32mKPSS - The series is {"not " if p_value < 0.05 else ""}Stationary\u001b[0m')


def build_decomposition_results(df):
    """
        A function that returns the trend, seasonality and residual values for multiplicative and
        additive model.
        df -> DataFrame
    """
    logger.info("Building result for seasonal decomposition model")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_pandas.iloc[:, 0] = pd.to_datetime(df_pandas.iloc[:, 0])
    df_pandas = df_pandas.set_index(df_pandas.columns[0]).asfreq('D').dropna()

    # Find length of time period to decide right outlier algorithm
    length_year = len(df_pandas.index) // 365.25

    logger.info("Time-series data length: %.2f years", length_year)

    if length_year >= 2.0:
        # Building Additive and Multiplicative time series models
        # In a multiplicative time series, the components multiply together to make the time series.
        # If there is an increasing trend, the amplitude of seasonal activity increases.
        # Everything becomes more exaggerated. This is common for web traffic.

        # In an additive time series, the components add together to make the time series.
        # If there is an increasing trend, we still see roughly the same size peaks and troughs
        # throughout the time series. This is often seen in indexed time series where the absolute value is
        # growing but changes stay relative.

        logger.info("Time-series data is more than 2 years")

        decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='additive',
                                                      extrapolate_trend='freq')
        residuals_add = get_residuals(decomposition_add)

        decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='multiplicative',
                                                      extrapolate_trend='freq')
        residuals_mul = get_residuals(decomposition_mul)

        # Get ACF values for both Additive and Multiplicative models
        # Calculate Sum of Squares of the ACF for both models
        ssacf_add: float = get_ssacf(residuals_add, type='Additive')
        ssacf_mul: float = get_ssacf(residuals_mul, type='Multiplicative')

        if ssacf_add < ssacf_mul:
            logger.info("Using Additive model for seasonal decomposition.")
            df_reconstructed = pd.concat([decomposition_add.seasonal, decomposition_add.trend,
                                          decomposition_add.resid, decomposition_add.observed], axis=1)
            df_reconstructed.columns = ['seasonal', 'trend', 'residuals', 'actual_values']
            return df_reconstructed
        else:
            logger.info("Using Multiplicative model for seasonal decomposition.")
            df_reconstructed = pd.concat([decomposition_mul.seasonal, decomposition_mul.trend,
                                          decomposition_mul.resid, decomposition_mul.observed], axis=1)
            df_reconstructed.columns = ['seasonal', 'trend', 'residuals', 'actual_values']
            return df_reconstructed
    else:
        logger.info("Data is less than 2 years.")
        print("Data is less than 2 years. No seasonal decomposition")


def build_outliers_plot_moving_average(df: pd.DataFrame) -> plt:
    """
     Show outliers using Moving Average and Z-score algorithm.

     Args:
         df (pd.DataFrame): A Pandas DataFrame with time-series data.
             First column must be a date column ('YYYY-MM-DD')
             and last column should be a count/feature column.

     Returns:
         plt: A plot with detected outliers.
     """

    logging.info("Plotting outliers using Moving Average method")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Calculate optimal window size
    optimal_window_size = calculate_optimal_window_size(df_pandas)

    # Ensure the DataFrame is indexed correctly
    if not isinstance(df_pandas.index, pd.DatetimeIndex):
        df_pandas = df_pandas.set_index(pd.to_datetime(df_pandas.iloc[:, 0]))

    # Calculate moving average
    df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])
    df1 = df_pandas.copy()
    df1['moving_average'] = df_pandas.iloc[:, -1].rolling(window=optimal_window_size).mean()

    # Call Z-score algorithm to detect dispersion
    z_scores = anomaly_zscore(df1['moving_average'])
    outliers = df1[np.abs(z_scores) > 2]

    if outliers.empty:
        print("No outlier detected using Moving Average method")
        return
    else:
        return_outliers = outliers.iloc[:, :2]
        return_outliers.reset_index(drop=True, inplace=True)
        print("Outlier detection using Moving Average method completed")
        print("Outliers:", return_outliers)

        # Plot the data
        plt.figure(figsize=(20, 8))
        plt.plot(df_pandas.iloc[:, -1], label='Original Data')
        plt.plot(df1['moving_average'], label='Moving Average')

        # Highlight outliers
        plt.scatter(outliers.index, outliers.iloc[:, 1], color='green', label='Outliers')
        plt.legend()
        logging.info("Completed outliers plotting using Moving Average method")


def build_outliers_plot_classic(df) -> plt:
    """
        Show outliers in a time-series dataset through Classical Seasonal Decomposition

        Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

        Returns:
            plot: A plot with detected outliers.
        """

    logging.info("Building outlier plot using classical seasonal decomposition.")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_pandas.iloc[:, 0] = df_pandas.iloc[:, 0].apply(pd.to_datetime)
    df_pandas = df_pandas.set_index(df_pandas.columns[0]).dropna()

    decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                  model='additive',
                                                  extrapolate_trend='freq')
    residuals_add = get_residuals(decomposition_add)

    decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                  model='multiplicative',
                                                  extrapolate_trend='freq')
    residuals_mul = get_residuals(decomposition_mul)

    # Get ACF values for both Additive and Multiplicative models

    # Calculate Sum of Squares of the ACF for both models
    ssacf_add: float = get_ssacf(residuals_add, type='Additive')
    ssacf_mul: float = get_ssacf(residuals_mul, type='Multiplicative')

    if ssacf_add < ssacf_mul:
        print("Additive Model")
        is_outlier = anomaly_mad(residuals_add)
        df_outliers = df_pandas[is_outlier]

    else:
        print("Multiplicative Model")
        is_outlier = anomaly_mad(residuals_mul)
        df_outliers = df_pandas[is_outlier]

    print("Outliers:", df_outliers)
    logging.info("Completing outlier detection using classical seasonal decomposition")

    # Plot the data
    plt.figure(figsize=(20, 8))
    plt.plot(df_pandas.iloc[:, -1], label='Original Data')

    # Highlight outliers
    plt.scatter(df_outliers.index, df_outliers.iloc[:, -1], color='red', label='Outliers')
    plt.legend()

    logging.info("Completing outlier plot using classical seasonal decomposition.")


def build_outliers_plot_stl(df) -> plt:
    """
    Show outliers in a time-series dataset through Seasonal-Trend Decomposition using LOESS (STL)

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        plot: A plot with detected outliers.
    """

    logging.info("Starting outlier detection using STL")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_stl = df_pandas.copy()
    df_stl.iloc[:, 0] = df_stl.iloc[:, 0].apply(pd.to_datetime)
    df_stl = df_stl.set_index(df_stl.columns[0]).dropna()

    # Ensure the datetime index is unique (no duplicate dates)
    if df_stl.index.is_unique:
        # Find the time frequency (daily, weekly etc.) and length of the index column
        inferred_frequency = df_stl.index.inferred_freq
        logging.info("Time frequency: %s", inferred_frequency)

        length_index = len(df_stl.index)
        logging.info("Length of time index: %.2f", length_index)

        # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition

        # Set parameter for Week check
        regex_week_check = r'[W-Za-z]'

        match inferred_frequency:
            case 'H' if length_index >= 17520:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # hour level time-series.")
                detected_period = 24  # Hourly seasonality
            case 'D' if length_index >= 730:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # day level time-series.")
                detected_period = 365  # Yearly seasonality
            case 'B' if length_index >= 520:
                # logging.info("Using seasonal trend decomposition for outlier detection in business
                # day level time-series.")
                detected_period = 365  # Yearly seasonality
            case 'MS' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                detected_period = 12
            case 'M' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                detected_period = 12
            case 'Q' if length_index >= 8:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # quarter level time-series.")
                detected_period = 4  # Quarterly seasonality
            case 'A' if length_index >= 2:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # annual level time-series.")
                detected_period = 1  # Annual seasonality
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    detected_period = 52  # Week level seasonality
                else:
                    # If less than 2 years of data, Use Inter Quartile Range (IQR) method
                    logging.info("Less than 2 years of data - Using IQR method for outlier detection")
                    return build_iqr_plot(df_pandas)
        return generate_outlier_plot_stl(df_stl, detected_period)
    else:
        print("Duplicate date index values. Check your data.")


def generate_outlier_plot_stl(df, detected_period) -> plt:
    """
    Generate outlier plot for time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
    MSTL can model seasonality which changes with time.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

    Returns:
        plt: A seasonal plot based on MSTL
    """

    derived_seasonal = detected_period + ((detected_period % 2) == 0)  # Ensure odd
    print("Detected Period: ", detected_period)
    print("Derived Seasonal: ", derived_seasonal)

    # Initializing df_outliers to avoid undefined usage
    df_outliers = pd.DataFrame()

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = STL(df_box['transformed_data'], seasonal=derived_seasonal, period=detected_period).fit()

    result_add = STL(df.iloc[:, -1], seasonal=derived_seasonal, period=detected_period).fit()

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) > np.var(result_add.resid):
        logging.info("Multiplicative model detected")
        type = 'multiplicative'
        df_outliers = generate_outliers_stl(df, type, derived_seasonal, detected_period)
    else:
        logging.info("Additive model detected")
        type = 'additive'
        df_outliers = generate_outliers_stl(df, type, derived_seasonal, detected_period)

    print("Outliers:", df_outliers)
    logging.info("Completing outlier detection using STL")

    plt.figure(figsize=(10, 4))
    plt.plot(df)
    plt.scatter(df_outliers.index, df_outliers.iloc[:, -1], color='r', marker='D')

    return plt


def build_seasonal_plot_stl(df) -> plt:
    """
    Build a seasonal plot for time-series dataset using Seasonal-Trend decomposition using Loess (STL).

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """
    logging.info("Starting outlier detection using STL")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_stl = df_pandas.copy()
    # Ensure the DataFrame is indexed correctly
    if not isinstance(df_stl.index, pd.DatetimeIndex):
        df_stl = df_stl.set_index(pd.to_datetime(df_stl.iloc[:, 0])).dropna()

    # Ensure the datetime index is unique (no duplicate dates)
    if df_stl.index.is_unique:
        # Find the time frequency (daily, weekly etc.) and length of the index column
        inferred_frequency = df_stl.index.inferred_freq
        logging.info("Time frequency: %s", inferred_frequency)

        length_index = len(df_stl.index)
        logging.info("Length of time index: %.2f", length_index)

        # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition

        # Set parameter for Week check
        regex_week_check = r'[W-Za-z]'

        match inferred_frequency:
            case 'H' if length_index >= 17520:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # hour level time-series.")
                detected_period = 24  # Hourly seasonality
            case 'D' if length_index >= 730:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # day level time-series.")
                detected_period = 365  # Yearly seasonality
            case 'B' if length_index >= 520:
                # logging.info("Using seasonal trend decomposition for outlier detection in business
                # day level time-series.")
                detected_period = 365  # Yearly seasonality
            case 'MS' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                detected_period = 12
            case 'M' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                detected_period = 12
            case 'Q' if length_index >= 8:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # quarter level time-series.")
                detected_period = 4  # Quarterly seasonality
            case 'A' if length_index >= 2:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # annual level time-series.")
                detected_period = 1  # Annual seasonality
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    detected_period = 52  # Week level seasonality
                else:
                    # If less than 2 years of data, Use Month-wise box plot method
                    logging.info("Less than 2 years of data - Month-wise box plot method")
                    return build_monthwise_plot(df_pandas)
        return generate_seasonal_plot_stl(df_stl, detected_period)
    else:
        print("Duplicate date index values. Check your data.")


def generate_seasonal_plot_stl(df, detected_period) -> plt:
    """
        Build a seasonal plot for time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
        MSTL can model seasonality which changes with time.

        Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

        Returns:
            plt: A seasonal plot based on MSTL
        """

    derived_seasonal = detected_period + ((detected_period % 2) == 0)  # Ensure odd
    print("Detected Period: ", detected_period)
    print("Derived Seasonal: ", derived_seasonal)

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = STL(df_box['transformed_data'], seasonal=derived_seasonal, period=detected_period).fit()
    result_add = STL(df.iloc[:, -1], seasonal=derived_seasonal, period=detected_period).fit()

    plt.rc("figure", figsize=(14, 16))
    plt.rc("font", size=12)

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) > np.var(result_add.resid):
        logging.info("Multiplicative model detected")
        result_mul.plot()
    else:
        logging.info("Additive model detected")
        result_add.plot()

    logging.info("Completing seasonal decomposition plot using STL")


def build_outliers_plot_mstl(df) -> plt:
    """
    Show outliers in a time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
    MSTL can model seasonality which changes with time.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        plot: A plot with detected outliers.
    """

    logging.info("Starting outlier detection using MSTL")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_mstl = df_pandas.copy()
    df_mstl.iloc[:, 0] = df_mstl.iloc[:, 0].apply(pd.to_datetime)
    df_mstl = df_mstl.set_index(df_mstl.columns[0]).dropna()

    # Ensure the datetime index is unique (no duplicate dates)
    if df_mstl.index.is_unique:
        # Find the time frequency (daily, weekly etc.) and length of the index column
        inferred_frequency = df_mstl.index.inferred_freq
        logging.info("Time frequency: %s", inferred_frequency)

        length_index = len(df_mstl.index)
        logging.info("Length of time index: %.2f", length_index)

        # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition
        # Set parameter for Week check
        regex_week_check = r'[W-Za-z]'

        match inferred_frequency:
            case 'H' if length_index >= 17520:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # hour level time-series.")
                period_hourly = 24
                period_weekly = period_hourly * 7
                derived_period = (period_hourly, period_weekly)  # Daily and Weekly Seasonality
            case 'D' if length_index >= 730:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # day level time-series.")
                period_weekly = 7
                period_yearly = 365
                derived_period = (period_weekly, period_yearly)  # Weekly and Yearly seasonality
            case 'B' if length_index >= 520:
                # logging.info("Using seasonal trend decomposition for outlier detection in business
                # day level time-series.")
                period_weekly = 5
                period_yearly = 365
                derived_period = (period_weekly, period_yearly)  # Weekly and Yearly seasonality
            case 'MS' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly  # Monthly seasonality
            case 'M' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly  # Monthly seasonality
            case 'Q' if length_index >= 8:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # quarter level time-series.")
                period_quarterly = 4
                period_yearly = 12
                derived_period = (period_quarterly, period_yearly)  # Quarterly and Yearly seasonality
            case 'A' if length_index >= 2:
                # logging.info("Using seasonal trend decomposition for outlier detection in
                # annual level time-series.")
                derived_period = 1  # Annual seasonality
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    derived_period = 52  # Week level seasonality
                else:
                    # If less than 2 years of data, Use Inter Quartile Range (IQR) method
                    logging.info("Less than 2 years of data - Using IQR method for outlier detection")
                    return build_iqr_plot(df_pandas)
        return generate_outlier_plot_mstl(df_mstl, derived_period)
    else:
        print("Duplicate date index values. Check your data.")


def generate_outlier_plot_mstl(df, derived_period) -> plt:
    """
     Generate outlier plot for time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
     MSTL can model seasonality which changes with time.

     Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

    Returns:
            plt: A seasonal plot based on MSTL
    """

    logging.info("Derived Period: %s", derived_period)

    # Initializing df_outliers to avoid undefined usage
    df_outliers = pd.DataFrame()

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = MSTL(df_box['transformed_data'], periods=derived_period).fit()
    result_add = MSTL(df.iloc[:, -1], periods=derived_period).fit()

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) > np.var(result_add.resid):
        logging.info("Multiplicative model detected")
        type = 'multiplicative'
        df_outliers = generate_outliers_mstl(df, type, derived_period)
    else:
        logging.info("Additive model detected")
        type = 'additive'
        df_outliers = generate_outliers_mstl(df, type, derived_period)

    print("Outliers:", df_outliers)
    logging.info("Completing outlier detection using MSTL")

    plt.figure(figsize=(10, 4))
    plt.plot(df)
    plt.scatter(df_outliers.index, df_outliers.iloc[:, -1], color='r', marker='D')

    return plt


def build_seasonal_plot_mstl(df) -> plt:
    """
    Build a seasonal plot for time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
    MSTL can model seasonality which changes with time.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """
    logging.info("Starting outlier detection using MSTL")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_mstl = df_pandas.copy()
    # Ensure the DataFrame is indexed correctly
    if not isinstance(df_mstl.index, pd.DatetimeIndex):
        df_mstl = df_mstl.set_index(pd.to_datetime(df_mstl.iloc[:, 0])).dropna()

    # Ensure the datetime index is unique (no duplicate dates)
    if df_mstl.index.is_unique:
        # Find the time frequency (daily, weekly etc.) and length of the index column
        inferred_frequency = df_mstl.index.inferred_freq
        logging.info("Time frequency: %s", inferred_frequency)

        length_index = len(df_mstl.index)
        logging.info("Length of time index: %.2f", length_index)

        # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition
        # Set parameter for Week check
        regex_week_check = r'[W-Za-z]'

        match inferred_frequency:
            case 'H' if length_index >= 17520:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # hour level time-series.")
                period_hourly = 24
                period_weekly = period_hourly * 7
                derived_period = (period_hourly, period_weekly)  # Daily and Weekly Seasonality
            case 'D' if length_index >= 730:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # day level time-series.")
                period_weekly = 7
                period_yearly = 365
                derived_period = (period_weekly, period_yearly)  # Weekly and Yearly seasonality
            case 'B' if length_index >= 520:
                # logging.info("Using seasonal trend decomposition for outlier detection in business
                # day level time-series.")
                period_weekly = 5
                period_yearly = 365
                derived_period = (period_weekly, period_yearly)   # Weekly and Yearly seasonality
            case 'MS' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly           # Monthly seasonality
            case 'M' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly           # Monthly seasonality
            case 'Q' if length_index >= 8:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # quarter level time-series.")
                period_quarterly = 4
                period_yearly = 12
                derived_period = (period_quarterly, period_yearly)    # Quarterly and Yearly seasonality
            case 'A' if length_index >= 2:
                # logging.info("Using seasonal trend decomposition for outlier detection in
                # annual level time-series.")
                derived_period = 1  # Annual seasonality
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    derived_period = 52  # Week level seasonality
                else:
                    # If less than 2 years of data, Use Month-Wise or Inter Quartile Range (IQR) method
                    logging.info("Less than 2 years of data - Use Month-wise or Moving Average Method")
                    logging.info("Default - Using Month-wise box plot")
                    return build_monthwise_plot(df_pandas)
        return generate_seasonal_plot_mstl(df_mstl, derived_period)
    else:
        print("Duplicate date index values. Check your data.")


def generate_seasonal_plot_mstl(df, derived_period) -> plt:
    """
        Build a seasonal plot for time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
        MSTL can model seasonality which changes with time.

        Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

        Returns:
            plt: A seasonal plot based on MSTL
        """

    logging.info("Derived Period: %s", derived_period)

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = MSTL(df_box['transformed_data'], periods=derived_period).fit()
    result_add = MSTL(df.iloc[:, -1], periods=derived_period).fit()

    plt.rc("figure", figsize=(14, 16))
    plt.rc("font", size=12)

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) > np.var(result_add.resid):
        logging.info("Multiplicative model detected")
        result_mul.plot()
    else:
        logging.info("Additive model detected")
        result_add.plot()

    logging.info("Completing seasonal decomposition plot using MSTL")


def build_outliers_plot_esd(df) -> plt:
    """
    In this method, time series anomalies are detected using the Generalized ESD algorithm.
    The generalized ESD (Extreme Studentized Deviate) test is used to detect one or more outliers
    in a univariate data set that follows an approximately Normal distribution.
    # http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    Arguments:
        df: Pandas dataframe
    Outputs:
        plt: Outliers plot (detected Generalized ESD anomalies)
    """

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_esd = df.toPandas()
    else:
        df_esd = df

    # Check for normality using the Shapiro-Wilk test to decide about right ESD method

    stat, p = shapiro(df_esd.iloc[:, -1])
    logging.info('Testing for Normality - Shapiro-Wilk Test Results:')
    logger.info("Statistic: %.3f", stat)
    logger.info('p-value: %.3f', p)

    # Setting Significance level default to 0.05
    alpha = 0.05

    # Decide right ESD method based on data distribution
    if p > alpha:
        logging.info("Data Likely Normally Distributed - Using Generalized ESD Method")
        # Call Generalized ESD function to generate outliers. Hybrid is set to True to use
        # Median & Median Absolute Deviation (MAD) else it would use the Mean & Standard Deviation of the residual.
        return_outliers = generate_outliers_generalized_esd(df_esd, hybrid=False)
        if return_outliers is None:
            logging.info("No outlier detected by Generalized ESD Method")
            return
        else:
            print("Outliers detected by Generalized ESD Method:")
            df_outliers = return_outliers.iloc[:, :2]
            print(df_outliers)
    else:
        print("Data Not Normally Distributed - Using Sesonal ESD Method")
        # Call Seasonal ESD function to generate outliers. Hybrid is set to True to use
        # Median & Median Absolute Deviation (MAD) else it would use the Mean & Standard Deviation of the residual.
        return_outliers = generate_outliers_seasonal_esd(df_esd, hybrid=True)
        if return_outliers is None:
            logging.info("No outlier detected by Seasonal ESD Method")
            return
        else:
            print("Outliers detected by Seasonal ESD Method:")
            df_outlier = return_outliers.iloc[:, :2]
            print(df_outlier)
            # Create df_match dataset to detect matching indices in the original dataframe and plot
            df_match = df_esd.copy()

            # Get matching indices in df_match and plot
            matching_indices = df_match.index.isin(df_outlier.index)

            # Mark outlier rows in df_match dataframe as outliers
            df_match['outlier'] = False
            df_match.loc[matching_indices, 'outlier'] = True

            # Plot the outliers
            plt.figure(figsize=(24, 10))
            df_plot = df_match.copy()
            df_plot['Month-Year'] = pd.to_datetime(df_plot.iloc[:, 0]).dt.to_period('M')
            df_plot['Month-Year'] = df_plot['Month-Year'].astype(str)
            df_plot['Count'] = pd.to_numeric(df_plot.iloc[:, 1])

            # Highlight outliers in the plot
            sns.scatterplot(data=df_plot, x='Month-Year', y='Count', hue='outlier').set_title("Seasonal ESD Anomalies")
            plt.show()