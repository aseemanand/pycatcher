import logging
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from .catch import get_residuals, get_ssacf

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


def build_seasonal_plot(df):
    """
    Build seasonal plot for a given dataframe
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
        # throughout the time series. This is often seen in indexed time series where the
        # absolute value is growing but changes stay relative.

        decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='additive', extrapolate_trend='freq')
        residuals_add = get_residuals(decomposition_add)

        decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='multiplicative', extrapolate_trend='freq')
        residuals_mul = get_residuals(decomposition_mul)

        # Get ACF values for both Additive and Multiplicative models

        ssacf_add = get_ssacf(residuals_add, df_pandas)
        ssacf_mul = get_ssacf(residuals_mul, df_pandas)

        # print('ssacf_add:', ssacf_add)
        # print('ssacf_mul:', ssacf_mul)

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
        Build IQR plot for a given dataframe
            Args:
                 df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                                   and the second/last column should be the feature (count).
        """

    logger.info("Building IQR plot to see outliers")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the last column is numeric
    df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])

    sns.boxplot(x=df_pandas.iloc[:, -1], showmeans=True)
    plt.show()


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

    df_pandas['Month-Year'] = pd.to_datetime(df_pandas.iloc[:, 0]).dt.to_period('M')
    df_pandas['Count'] = pd.to_numeric(df_pandas.iloc[:, 1])
    plt.figure(figsize=(30, 4))
    sns.boxplot(x='Month-Year', y='Count', data=df_pandas).set_title("Month-wise Box Plot")
    plt.show()


def conduct_stationarity_check(df):

    """
    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        ADF statistics, Stationarity check (trend only). Time series are stationary if they
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

    logger.info("Building stationarity check")

    result = sm.tsa.stattools.adfuller(df_pandas.iloc[:, -1].values)

    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")


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

        decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='additive',extrapolate_trend='freq')
        residuals_add = get_residuals(decomposition_add)

        decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                      model='multiplicative',extrapolate_trend='freq')
        residuals_mul = get_residuals(decomposition_mul)

        # Get ACF values for both Additive and Multiplicative models

        ssacf_add = get_ssacf(residuals_add, df_pandas)
        ssacf_mul = get_ssacf(residuals_mul, df_pandas)

        if ssacf_add < ssacf_mul:
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
        print ("Data is less than 2 years. No seasonal decomposition")