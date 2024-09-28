import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from pyod.models.mad import MAD


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

