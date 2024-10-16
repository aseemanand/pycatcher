import math
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Union
from pyod.models.mad import MAD
from sklearn.base import BaseEstimator
from statsmodels.tsa.stattools import acf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the second column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the rows that are considered outliers.
    """

    logging.info("Detecting outliers using the IQR method.")

    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the second column
    q1 = df.iloc[:, 1].quantile(0.25)
    q3 = df.iloc[:, 1].quantile(0.75)

    # Calculate the Inter Quartile Range (IQR)
    iqr = q3 - q1

    # Identify outliers
    outliers = df[((df.iloc[:, 1] < (q1 - 1.5 * iqr)) | (df.iloc[:, 1] > (q3 + 1.5 * iqr)))]

    logging.info(f"Outliers detected: {len(outliers)} rows.")

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

    logging.info("Detecting outliers using the MAD method.")

    # Reshape residuals from the fitted model
    residuals = model_type.resid.values.reshape(-1, 1)

    # Fit the MAD outlier detection model
    mad = MAD().fit(residuals)

    # Identify outliers using MAD labels (1 indicates an outlier)
    is_outlier = mad.labels_ == 1

    logging.info(f"Outliers detected by MAD")

    return is_outlier


def get_residuals(model_type: BaseEstimator) -> np.ndarray:
    """
    Get the residuals of a fitted model, removing any NaN values.

    Args:
        model_type (BaseEstimator): A fitted model object that has the attribute `resid`,
                                    representing the residuals of the model.

    Returns:
        np.ndarray: An array of residuals with NaN values removed.
    """

    logging.info("Extracting residuals and removing NaN values.")

    # Extract residuals from the model and remove NaN values
    residuals = model_type.resid.values
    residuals_cleaned = residuals[~np.isnan(residuals)]

    logging.info(f"Number of residuals after NaN removal: {len(residuals_cleaned)}")

    return residuals_cleaned


def sum_of_squares(array: np.ndarray) -> float:
    """
    Calculates the sum of squares of a NumPy array of any shape.

    Args:
        array (np.ndarray): A NumPy array of any shape.

    Returns:
        float: The sum of squares of the array elements.
    """

    logging.info("Calculating the sum of squares.")

    # Flatten the array to a 1D array
    flattened_array = array.flatten()

    # Calculate the sum of squares of the flattened array
    sum_of_squares_value = np.sum(flattened_array ** 2)

    logging.info(f"Sum of squares calculated: {sum_of_squares_value:.2f}")

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

    logging.info("Calculating the sum of squares of the ACF of residuals.")

    # Calculate the number of lags based on the square root of the data length
    range_var = len(df_pandas.index)
    nlags = math.isqrt(range_var) + 45

    # Compute the ACF of the residuals
    acf_array = acf(residuals, nlags=nlags, fft=True)

    # Calculate the sum of squares of the ACF values
    ssacf = sum_of_squares(acf_array)

    logging.info(f"Sum of squares of ACF: {ssacf:.2f}")

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

    logging.info("Detecting today's outliers.")

    # Get the DataFrame of outliers from detect_outliers and select the latest row
    df_outliers = detect_outliers(df)
    df_last_outlier = df_outliers.tail(1)

    # Extract the latest outlier's date
    last_outlier_date = df_last_outlier.index[-1].date().strftime('%Y-%m-%d')

    # Get the current date
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Check if the latest outlier occurred today
    if last_outlier_date == current_date:
        logging.info("Outliers detected today.")
        return df_last_outlier
    else:
        logging.info("No outliers detected today.")
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

    logging.info("Detecting the latest outliers.")

    df_outliers = detect_outliers(df)
    df_latest_outlier = df_outliers.tail(1)

    logging.info("Detected the latest outlier!")

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

    logging.info("Starting outlier detection.")

    # Creating a shallow copy of Pandas dataframe to use for seasonal trend
    df_pandas = df.copy(deep=False)

    # Calculate the length of the time period in years
    length_year: float = len(df_pandas.index) / 365.25

    logging.info(f"Time-series data in years: {length_year:.2f}")

    # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition
    if length_year >= 2.0:
        logging.info("Using seasonal trend decomposition for outlier detection.")
        return _decompose_and_detect(df_pandas)

    # If less than 2 years of data, use Inter Quartile Range (IQR) method
    else:
        logging.info("Using IQR method for outlier detection.")
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

    logging.info("Decomposing time-series for additive and multiplicative models.")

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
        logging.info("Using the additive model for outlier detection.")
        is_outlier = anomaly_mad(decomposition_add)
    else:
        logging.info("Using the multiplicative model for outlier detection.")
        is_outlier = anomaly_mad(decomposition_mul)

    # Use the aligned boolean Series as the indexer
    df_outliers = df_pandas[is_outlier]

    if df_outliers.empty:
        logging.info("No outliers found.")
        return "No outliers found."

    logging.info(f"Outliers detected: {len(df_outliers)} rows.")

    return df_outliers


def _detect_outliers_iqr(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        pd.DataFrame: A DataFrame containing the detected outliers.
    """

    logging.info("Detecting outliers using the IQR method.")

    # Ensure the second column is numeric
    df_pandas.iloc[:, 1] = pd.to_numeric(df_pandas.iloc[:, 1])

    # Detect outliers using the IQR method
    df_outliers: pd.DataFrame = find_outliers_iqr(df_pandas)

    logging.info(f"Outliers detected using IQR: {len(df_outliers)} rows.")

    return df_outliers