import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Union
from pyod.models.mad import MAD
from sklearn.base import BaseEstimator
from statsmodels.tsa.stattools import acf


def find_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the second column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the rows that are considered outliers.
    """

    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the second column
    q1 = df.iloc[:, 1].quantile(0.25)
    q3 = df.iloc[:, 1].quantile(0.75)

    # Calculate the Inter Quartile Range (IQR)
    iqr = q3 - q1

    # Identify outliers
    outliers = df[((df.iloc[:, 1] < (q1 - 1.5 * iqr)) | (df.iloc[:, 1] > (q3 + 1.5 * iqr)))]

    return outliers


def anomaly_mad(model_type: BaseEstimator) -> pd.DataFrame:
    """
    Detect outliers using the Median Absolute Deviation (MAD) method.
    MAD is a statistical measure that quantifies the dispersion or variability of a dataset.

    Args:
        model_type (BaseEstimator): A model object that has been fitted to the data, containing residuals.

    Returns:
        pd.DataFrame: A DataFrame containing the rows identified as outliers.
    """

    # Reshape residuals from the fitted model
    residuals = model_type.resid.values.reshape(-1, 1)

    # Fit the MAD outlier detection model
    mad = MAD().fit(residuals)

    # Identify outliers using MAD labels (1 indicates an outlier)
    is_outlier = mad.labels_ == 1

    return(is_outlier)


def get_residuals(model_type: BaseEstimator) -> np.ndarray:
    """
    Get the residuals of a fitted model, removing any NaN values.

    Args:
        model_type (BaseEstimator): A fitted model object that has the attribute `resid`,
                                    representing the residuals of the model.

    Returns:
        np.ndarray: An array of residuals with NaN values removed.
    """

    # Extract residuals from the model and remove NaN values
    residuals = model_type.resid.values
    residuals_cleaned = residuals[~np.isnan(residuals)]

    return residuals_cleaned


def sum_of_squares(array: np.ndarray) -> float:
    """
    Calculates the sum of squares of a NumPy array of any shape.

    Args:
        array (np.ndarray): A NumPy array of any shape.

    Returns:
        float: The sum of squares of the array elements.
    """

    # Flatten the array to a 1D array
    flattened_array = array.flatten()

    # Calculate the sum of squares of the flattened array
    sum_of_squares_value = np.sum(flattened_array ** 2)

    return sum_of_squares_value


def get_ssacf(residuals: np.ndarray, df_pandas: pd.DataFrame) -> float:
    """
    Get the sum of squares of the Auto Correlation Function (ACF) of the residuals.

    Args:
        residuals (np.ndarray): A NumPy array containing the residuals.
        df_pandas (pd.DataFrame): A pandas DataFrame containing the data.

    Returns:
        float: The sum of squares of the ACF of the residuals.
    """

    # Calculate the number of lags based on the square root of the data length
    range_var = len(df_pandas.index)
    nlags = math.isqrt(range_var) + 45

    # Compute the ACF of the residuals
    acf_array = acf(residuals, nlags=nlags, fft=True)

    # Calculate the sum of squares of the ACF values
    ssacf = sum_of_squares(acf_array)

    return ssacf


def detect_outliers_today(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect the outliers detected today using the anomaly_mad method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the second/last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing today's outliers if detected.
        str: A message indicating no outliers were found today.
    """

    # Get the DataFrame of outliers from detect_outliers and select the latest row
    df_outliers = detect_outliers(df)
    df_last_outlier = df_outliers.tail(1)

    # Extract the latest outlier's date
    last_outlier_date = df_last_outlier.index[-1].date().strftime('%Y-%m-%d')

    # Get the current date
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Check if the latest outlier occurred today
    if last_outlier_date == current_date:
        return df_last_outlier
    else:
        return "No Outliers Today!"


def detect_outliers_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the last outliers detected using the detect_outlier method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the second column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the latest detected outlier.
    """
    df_outliers = detect_outliers(df)
    df_latest_outlier = df_outliers.tail(1)
    return df_latest_outlier


def detect_outliers(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect outliers in a time-series dataset using Seasonal Trend Decomposition
    when there is at least 2 years of data, otherwise use Inter Quartile Range (IQR) for smaller timeframes.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and Second/last column should be a count/feature column.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """
    # Creating a shallow copy of Pandas dataframe to use for seasonal trend
    df_pandas = df.copy(deep=False)

    # Calculate the length of the time period in years
    length_year: float = len(df_pandas.index) / 365.25
    print(f"Time-series data in years: {length_year:.2f}")

    # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition
    if length_year >= 2.0:
        return _decompose_and_detect(df_pandas)

    # If less than 2 years of data, use Inter Quartile Range (IQR) method
    else:
        return _detect_outliers_iqr(df)


def _decompose_and_detect(df_pandas: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Helper function to apply Seasonal Trend Decomposition and detect outliers using
    both additive and multiplicative models.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        str or pd.DataFrame: A message or a DataFrame with detected outliers.
    """

    # Ensure the first column is in datetime format and set it as index
    df_pandas.iloc[:, 0] = pd.to_datetime(df_pandas.iloc[:, 0])
    df_pandas = df_pandas.set_index(df_pandas.columns[0]).asfreq('D').dropna()

    # Decompose the series using both additive and multiplicative models
    decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1], model='additive')
    decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1], model='multiplicative')

    # Get residuals from both decompositions
    residuals_add: pd.Series = get_residuals(decomposition_add)
    residuals_mul: pd.Series = get_residuals(decomposition_mul)

    # Calculate Sum of Squares of the ACF for both models
    ssacf_add: float = get_ssacf(residuals_add, df_pandas)
    ssacf_mul: float = get_ssacf(residuals_mul, df_pandas)

    # Return the outliers detected by the model with the smaller ACF value
    if ssacf_add < ssacf_mul:
        print("Additive Model")
        is_outlier = anomaly_mad(decomposition_add)
    else:
        print("Multiplicative Model")
        is_outlier = anomaly_mad(decomposition_mul)

    # Use the aligned boolean Series as the indexer
    df_outliers = df_pandas[is_outlier]

    if df_outliers.empty:
        return "No outliers found."

    return df_outliers


def _detect_outliers_iqr(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        pd.DataFrame: A DataFrame containing the detected outliers.
    """
    # Ensure the second column is numeric
    df_pandas.iloc[:,1] = pd.to_numeric(df_pandas.iloc[:,1])

    # Detect outliers using the IQR method
    df_outliers: pd.DataFrame = find_outliers_iqr(df_pandas)
    print(f"Record Count: {len(df_outliers)}")
    return df_outliers

def plot_seasonal(res, axes, title):
    """
    Args:
        res: Model type result
        axes: An Axes typically has a pair of Axis Artists that define the data coordinate system, and include methods to add annotations like x- and y-labels, titles, and legends.
        title: Title of the plot

    """
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
    # Import the necessary libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert to Pandas dataframe for easy manipulation
    df_pandas = df.toPandas()

    # Ensure the first column is in datetime format and set it as index
    df_pandas.iloc[:, 0] = pd.to_datetime(df_pandas.iloc[:, 0])
    df_pandas = df_pandas.set_index(df_pandas.columns[0]).asfreq('D').dropna()

    # Find length of time period to decide right outlier algorithm
    length_year = len(df_pandas.index) // 365.25
    # print('Time-series data in years:',length_year)

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

        # print('ssacf_add:',ssacf_add)
        # print('ssacf_mul:',ssacf_mul)

        if ssacf_add < ssacf_mul:
            print("Additive Model")
            fig, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_add, axes, title="Additive")
        else:
            print("Multiplicative Model")
            fig, axes = plt.subplots(ncols=1, nrows=4, sharex=False, figsize=(30, 15))
            plot_seasonal(decomposition_mul, axes, title="Multiplicative")
    else:
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
    # Import the necessary libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert to Pandas dataframe for easy manipulation
    df_pandas = df.toPandas()
    df_pandas['Month-Year'] = pd.to_datetime(df_pandas.iloc[:, 0]).dt.to_period('M')
    df_pandas['Count'] = pd.to_numeric(df_pandas.iloc[:, 1])
    plt.figure(figsize=(30, 4))
    sns.boxplot(x='Month-Year', y='Count', data=df_pandas).set_title("Month-wise Box Plot")
    plt.show()

