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
    Detect outliers using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): A DataFrame containing the data. The first column should be the identifier,
                           and the second column should be the feature for which outliers are detected.

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


def anomaly_mad(model_type: BaseEstimator, df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the Median Absolute Deviation (MAD) method.
    MAD is a statistical measure that quantifies the dispersion or variability of a dataset.

    Args:
        model_type (BaseEstimator): A model object that has been fitted to the data, containing residuals.
        df (pd.DataFrame): A pandas DataFrame containing the data. The outliers will be selected from this DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the rows identified as outliers.
    """

    # Reshape residuals from the fitted model
    residuals = model_type.resid.values.reshape(-1, 1)

    # Fit the MAD outlier detection model
    mad = MAD().fit(residuals)

    # Identify outliers using MAD labels (1 indicates an outlier)
    is_outlier = mad.labels_ == 1

    # Select and return the rows corresponding to outliers
    outliers = df[is_outlier]

    return outliers


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
    Get the sum of squares of the autocorrelation function (ACF) of the residuals.

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


def get_outliers_today(model_type: BaseEstimator) -> Union[pd.DataFrame, str]:
    """
    Get the outliers detected today using the anomaly_mad method.

    Args:
        model_type (BaseEstimator): A fitted model object used to detect outliers.

    Returns:
        pd.DataFrame: A DataFrame containing today's outliers if detected.
        str: A message indicating no outliers were found today.
    """

    # Get the DataFrame of outliers from anomaly_mad and select the latest row
    df = anomaly_mad(model_type).tail(1)

    # Extract the latest outlier's date
    latest_outlier_date = df.index[-1].date().strftime('%Y-%m-%d')

    # Get the current date
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Check if the latest outlier occurred today
    if latest_outlier_date == current_date:
        return df

    return "No Outliers Today!"


def detect_outliers(df: pd.DataFrame) -> str | pd.DataFrame:
    """
    Detect outliers in a time-series dataset using Seasonal Trend Decomposition
    if there is at least 2 years of data, or Interquartile Range (IQR) for shorter data.

    Args:
        df: A Pandas DataFrame with time-series data.
            It must contain 'session_start_dt' and 'count_clickstream' columns.

    Returns:
        str or pd.DataFrame: A message or a DataFrame with detected outliers.
    """

    # Calculate the length of the time period in years
    length_year: float = len(df.index) / 365.25
    print(f"Time-series data in years: {length_year:.2f}")

    # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition
    if length_year >= 2.0:
        return _decompose_and_detect(df)

    # If less than 2 years of data, use Inter Quartile Range (IQR) method
    else:
        return _detect_outliers_iqr(df)


def _decompose_and_detect(df_pandas: pd.DataFrame) -> str | pd.DataFrame:
    """
    Helper function to apply Seasonal Decomposition and detect outliers using
    both additive and multiplicative models.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        str or pd.DataFrame: A message or a DataFrame with detected outliers.
    """

    # Ensure the 'session_start_dt' column is in datetime format and set it as index
    df_pandas['session_start_dt'] = pd.to_datetime(df_pandas['session_start_dt'])
    df_pandas = df_pandas.set_index('session_start_dt').asfreq('D').dropna()

    # Decompose the series using both additive and multiplicative models
    decomposition_add = sm.tsa.seasonal_decompose(df_pandas['count_clickstream'], model='additive')
    decomposition_mul = sm.tsa.seasonal_decompose(df_pandas['count_clickstream'], model='multiplicative')

    # Get residuals from both decompositions
    residuals_add: pd.Series = get_residuals(decomposition_add)
    residuals_mul: pd.Series = get_residuals(decomposition_mul)

    # Calculate Sum of Squares of the ACF for both models
    ssacf_add: float = get_ssacf(residuals_add, df_pandas)
    ssacf_mul: float = get_ssacf(residuals_mul, df_pandas)

    # Return the outliers detected by the model with the smaller ACF value
    if ssacf_add < ssacf_mul:
        return get_outliers_today(decomposition_add)
    else:
        return get_outliers_today(decomposition_mul)


def _detect_outliers_iqr(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to detect outliers using the Interquartile Range (IQR) method.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        pd.DataFrame: A DataFrame containing the detected outliers.
    """
    # Ensure the 'count_clickstream' column is numeric
    df_pandas['count_clickstream'] = pd.to_numeric(df_pandas['count_clickstream'], errors='coerce')

    # Detect outliers using the IQR method
    df_outliers: pd.DataFrame = find_outliers_iqr(df_pandas)
    print(f"Record Count: {len(df_outliers)}")
    return df_outliers
