import logging
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.pycatcher.catch import get_residuals, get_ssacf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_seasonal(res, axes, title):
    """
    Args:
        res: Model type result
        axes: An Axes typically has a pair of Axis Artists that define the data coordinate system, and include methods to add annotations like x- and y-labels, titles, and legends.
        title: Title of the plot

    """

    logger.info(f"Plotting seasonal decomposition with title: {title}")

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


def build_plot(df):
    """
    Build plot for a given dataframe
        Args:
             df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                               and the second/last column should be the feature (count).
    """

    logger.info("Building time-series plot for seasonal decomposition.")

    # Convert to Pandas dataframe for easy manipulation
    df_pandas = df.toPandas()

    # Ensure the first column is in datetime format and set it as index
    df_pandas.iloc[:, 0] = pd.to_datetime(df_pandas.iloc[:, 0])
    df_pandas = df_pandas.set_index(df_pandas.columns[0]).asfreq('D').dropna()

    # Find length of time period to decide right outlier algorithm
    length_year = len(df_pandas.index) // 365.25

    logger.info(f"Time-series data length: {length_year:.2f} years")

    if length_year >= 2.0:

        # Building Additive and Multiplicative time series models
        # In a multiplicative time series, the components multiply together to make the time series.
        # If there is an increasing trend, the amplitude of seasonal activity increases.
        # Everything becomes more exaggerated. This is common for web traffic.

        # In an additive time series, the components add together to make the time series.
        # If there is an increasing trend, we still see roughly the same size peaks and troughs throughout the time series.
        # This is often seen in indexed time series where the absolute value is growing but changes stay relative.

        decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1], model='additive')
        residuals_add = get_residuals(decomposition_add)

        decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1], model='multiplicative')
        residuals_mul = get_residuals(decomposition_mul)

        # Get ACF values for both Additive and Multiplicative models

        ssacf_add = get_ssacf(residuals_add, df_pandas)
        ssacf_mul = get_ssacf(residuals_mul, df_pandas)

        # print('ssacf_add:', ssacf_add)
        # print('ssacf_mul:', ssacf_mul)

        if ssacf_add < ssacf_mul:
            logger.info("Using Additive model for seasonal decomposition.")
            fig, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_add, axes, title="Additive")
        else:
            logger.info("Using Multiplicative model for seasonal decomposition.")
            fig, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_mul, axes, title="Multiplicative")
    else:
        logger.info("Using boxplot since the data is less than 2 years.")
        df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])
        sns.boxplot(x=df_pandas.iloc[:, -1], showmeans=True)
        plt.show()


def build_monthwise_plot(df):
    """
        Build month-wise plot for a given dataframe
            Args:
                 df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                                   and the second/last column should be the feature (count).
    """

    logger.info("Building month-wise box plot.")

    # Convert to Pandas dataframe for easy manipulation
    df_pandas = df.toPandas()
    df_pandas['Month-Year'] = pd.to_datetime(df_pandas.iloc[:, 0]).dt.to_period('M')
    df_pandas['Count'] = pd.to_numeric(df_pandas.iloc[:, 1])
    plt.figure(figsize=(30, 4))
    sns.boxplot(x='Month-Year', y='Count', data=df_pandas).set_title("Month-wise Box Plot")
    plt.show()
