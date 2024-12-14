import logging
from typing import Union
import re as regex
import numpy as np
import pandas as pd
from pyod.models.mad import MAD
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from scipy import stats
from scipy.special import inv_boxcox


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# warnings.resetwarnings()


def check_and_convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """Checks if the first column of a DataFrame is in date format,
       and converts it to 'yyyy-mm-dd' format if necessary."""

    first_col_name = df.columns[0]

    # Check if the column is already in datetime format
    if pd.api.types.is_datetime64_any_dtype(df[first_col_name]):
        df[df.columns[0]] = df[df.columns[0]].apply(pd.to_datetime)
        df = df.set_index(df.columns[0]).dropna()
    else:
        try:
            # Attempt to convert the column to datetime
            df[df.columns[0]] = df[df.columns[0]].apply(pd.to_datetime)
            df = df.set_index(df.columns[0]).dropna()
        except ValueError:
            print("First column is not in a recognizable date format.")
    return df


def find_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the rows that are considered outliers.
    """

    logging.info("Detecting outliers using the IQR method.")

    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the last column
    q1 = df.iloc[:, -1].quantile(0.25)
    q3 = df.iloc[:, -1].quantile(0.75)

    # Calculate the Inter Quartile Range (IQR)
    iqr = q3 - q1

    # Identify outliers
    outliers = df[((df.iloc[:, -1] < (q1 - 1.5 * iqr)) | (df.iloc[:, -1] > (q3 + 1.5 * iqr)))]

    logging.info("Outliers detected: %d rows.", len(outliers))

    return outliers


def anomaly_mad(residuals: BaseEstimator) -> pd.DataFrame:
    """
    Detect outliers using the Median Absolute Deviation (MAD) method.
    MAD is a statistical measure that quantifies the dispersion or variability of a dataset.
    https://www.sciencedirect.com/science/article/abs/pii/S0022103113000668

    Args:
        residuals (BaseEstimator): Residuals from seasonal decomposition.

    Returns:
        pd.DataFrame: A DataFrame containing the rows identified as outliers.
    """

    logging.info("Detecting outliers using the MAD method.")

    # Reshape residuals from the fitted model
    if isinstance(residuals, np.ndarray):
        residuals = residuals.reshape(-1, 1)
    else:
        residuals = residuals.values.reshape(-1, 1)

    # Fit the MAD outlier detection model
    mad = MAD().fit(residuals)

    # Identify outliers using MAD labels (1 indicates an outlier)
    is_outlier = mad.labels_ == 1

    logging.info("Outliers detected by MAD!")

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

    logging.info("Number of residuals after NaN removal: %d", len(residuals_cleaned))

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

    logging.info("Sum of squares calculated: %.2f", sum_of_squares_value)

    return float(sum_of_squares_value)


def get_ssacf(residuals: np.ndarray) -> float:
    """
    Get the sum of squares of the Auto Correlation Function (ACF) of the residuals.

    Args:
        residuals (np.ndarray): A NumPy array containing the residuals.

    Returns:
        float: The sum of squares of the ACF of the residuals.
    """

    logging.info("Calculating the sum of squares of the ACF of residuals.")

    # Compute the ACF of the residuals
    acf_array = acf(residuals, fft=True)

    # Calculate the sum of squares of the ACF values
    ssacf = sum_of_squares(acf_array)

    logging.info("Sum of squares of ACF: %.2f", ssacf)

    return ssacf


def detect_outliers_today_classic(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect the outliers detected today using the anomaly_mad method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing today's outliers if detected.
        str: A message indicating no outliers were found today.
    """

    logging.info("Detecting today's outliers.")

    # Get the DataFrame of outliers from detect_outliers and select the latest row
    df_outliers = detect_outliers_classic(df)
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


def detect_outliers_latest_classic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the last outliers detected using the detect_outlier method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the latest detected outlier.
    """

    logging.info("Detecting the latest outliers.")

    df_outliers = detect_outliers_classic(df)
    df_latest_outlier = df_outliers.tail(1)

    logging.info("Detected the latest outlier!")

    return df_latest_outlier


def detect_outliers_classic(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect outliers in a time-series dataset using Seasonal Trend Decomposition
    when there is at least 2 years of data, otherwise use Inter Quartile Range (IQR) for smaller timeframes.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """

    logging.info("Starting outlier detection.")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the first column is in datetime format and set it as index
    df_pandas = check_and_convert_date(df_pandas)

    # Ensure the datetime index is unique (no duplicate dates)
    if df_pandas.index.is_unique:
        # Find the time frequency (daily, weekly etc.) and length of the index column
        inferred_frequency = df_pandas.index.inferred_freq
        logging.info("Time frequency: %s", inferred_frequency)

        length_index = len(df_pandas.index)
        logging.info("Length of time index: %.2f", length_index)

        # If the dataset contains at least 2 years of data, use Seasonal Trend Decomposition

        # Set parameter for Week check
        regex_week_check = r'[W-Za-z]'

        match inferred_frequency:
            case 'D' if length_index >= 730:
                logging.info("Using seasonal trend decomposition for for outlier detection in day level time-series.")
                df_outliers = decompose_and_detect(df_pandas)
                return df_outliers
            case 'B' if length_index >= 520:
                logging.info(
                    "Using seasonal trend decomposition for outlier detection in business day level time-series.")
                df_outliers = decompose_and_detect(df_pandas)
                return df_outliers
            case 'MS' if length_index >= 24:
                logging.info("Using seasonal trend decomposition for for outlier detection in month level time-series.")
                df_outliers = decompose_and_detect(df_pandas)
                return df_outliers
            case 'Q' if length_index >= 8:
                logging.info("Using seasonal trend decomposition for outlier detection in quarter level time-series.")
                df_outliers = decompose_and_detect(df_pandas)
                return df_outliers
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    df_outliers = decompose_and_detect(df_pandas)
                    return df_outliers
                else:
                    # If less than 2 years of data, use Inter Quartile Range (IQR) method
                    logging.info("Using IQR method for outlier detection.")
                    return detect_outliers_iqr(df_pandas)
    else:
        print("Duplicate date index values. Check your data.")


def decompose_and_detect(df_pandas: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Helper function to apply Seasonal Trend Decomposition and detect outliers using
    both additive and multiplicative models.

    Args:
        df_pandas (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        str or pd.DataFrame: A message or a DataFrame with detected outliers.
    """

    logging.info("Decomposing time-series for additive and multiplicative models.")

    # Decompose the series using both additive and multiplicative models
    decomposition_add = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                  model='additive',
                                                  extrapolate_trend='freq')
    decomposition_mul = sm.tsa.seasonal_decompose(df_pandas.iloc[:, -1],
                                                  model='multiplicative',
                                                  extrapolate_trend='freq')

    # Get residuals from both decompositions
    residuals_add: pd.Series = get_residuals(decomposition_add)
    residuals_mul: pd.Series = get_residuals(decomposition_mul)

    # Calculate Sum of Squares of the ACF for both models
    ssacf_add: float = get_ssacf(residuals_add)
    ssacf_mul: float = get_ssacf(residuals_mul)

    # Return the outliers detected by the model with the smaller ACF value
    if ssacf_add < ssacf_mul:
        logging.info("Using the additive model for outlier detection.")
        is_outlier = anomaly_mad(residuals_add)
    else:
        logging.info("Using the multiplicative model for outlier detection.")
        is_outlier = anomaly_mad(residuals_mul)

    # Use the aligned boolean Series as the indexer
    df_outliers = df_pandas[is_outlier]

    if df_outliers.empty:
        logging.info("No outliers found.")
        return "No outliers found."

    logging.info("Outliers detected: %d rows.", len(df_outliers))

    return df_outliers


def detect_outliers_iqr(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Helper function to detect outliers using the Inter Quartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing time-series data.

    Returns:
        pd.DataFrame: A DataFrame containing the detected outliers.
    """

    logging.info("Detecting outliers using the IQR method.")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Ensure the last column is numeric
    df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])

    # Detect outliers using the IQR method
    df_outliers: pd.DataFrame = find_outliers_iqr(df_pandas)

    if df_outliers.empty:
        logging.info("No outliers found.")
        return "No outliers found."

    logging.info("Outliers detected using IQR: %d rows.", len(df_outliers))

    return df_outliers


def calculate_rmse(df: pd.DataFrame, window_size: int) -> list:
    """
    Calculate RMSE for a given window size

    Args:
        df (pd.DataFrame): A Pandas DataFrame
        Last column should be a count/feature column.

    Returns:
        list: mean of RMSE
    """

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_index, test_index in tscv.split(df):
        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

        train_df['ma'] = train_df.iloc[:, -1].rolling(window=window_size).mean()
        test_df['ma'] = test_df.iloc[:, -1].rolling(window=window_size).mean()

        # Drop NaN values from the test dataframe
        test_df = test_df.dropna()

        # Ensure test_df is not empty
        if not test_df.empty:
            rmse = np.sqrt(mean_squared_error(test_df.iloc[:, -1], test_df['ma']))
            rmse_scores.append(rmse)
    return np.mean(rmse_scores) if rmse_scores else np.nan


def calculate_optimal_window_size(df: pd.DataFrame) -> str:
    """
    Calculate optimal window size for Moving Average. The window size determines the
    number of data points used to calculate the moving average. A larger window size
    results in a smoother moving average but with less responsiveness to short-term changes.
    A smaller window size results in a more responsive moving average but with more noise.
    The optimal window size depends on the business context and the goal of the analysis.

    Args:
        df (pd.DataFrame): A Pandas DataFrame
        Last column should be a count/feature column.

    Returns:
        str: optimal window size
    """

    logging.info("Starting optimal window size calculation")

    # Try different window sizes
    window_sizes = range(2, 21)
    rmse_values = []

    logging.info("Starting RMSE calculation")
    for window_size in window_sizes:
        rmse = calculate_rmse(df, window_size)
        rmse_values.append(rmse)
    logging.info("RMSE calculation completed")

    # Check if all rmse_values are NaN
    if np.all(np.isnan(rmse_values)):
        raise ValueError("All RMSE values are NaN. Check your data for issues.")

    # Find the window size with the lowest RMSE
    optimal_window_size = window_sizes[np.nanargmin(rmse_values)]
    logging.info("Optimal Window Size: %d", optimal_window_size)
    return optimal_window_size


def detect_outliers_moving_average(df: pd.DataFrame) -> str:
    """
     Detect outliers using Moving Average method.

     Args:
         df (pd.DataFrame): A Pandas DataFrame with time-series data.
          Last column should be a count/feature column

     Returns:
         str: A message with None found or with detected outliers.
     """

    logging.info("Starting outlier detection using Moving Average method")

    # Check whether the argument is Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        # Convert to Pandas dataframe for easy manipulation
        df_pandas = df.toPandas()
    else:
        df_pandas = df

    # Calculate optimal window size
    optimal_window_size = calculate_optimal_window_size(df_pandas)

    # Calculate moving average
    df_pandas.iloc[:, -1] = pd.to_numeric(df_pandas.iloc[:, -1])
    df1 = df_pandas.copy()
    df1['moving_average'] = df_pandas.iloc[:, -1].rolling(window=optimal_window_size).mean()

    # Set a threshold of 2 standard deviations from the moving average
    threshold = df1['moving_average'].std() * 2

    # Identify values that cross the threshold
    df1['above_threshold'] = df_pandas.iloc[:, -1] > (df1['moving_average'] + threshold)
    df1['below_threshold'] = df_pandas.iloc[:, -1] < (df1['moving_average'] - threshold)

    outliers = df1[(df1['above_threshold']) | (df1['below_threshold'])].dropna()
    return_outliers = outliers.iloc[:, :2]
    return_outliers.reset_index(drop=True, inplace=True)
    logging.info("Outlier detection using Moving Average method completed")
    return return_outliers


def detect_outliers_stl(df) -> Union[pd.DataFrame, str]:
    """
    Detect outliers in a time-series dataset through Seasonal-Trend Decomposition using LOESS (STL)

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
                    # If less than 2 years of data, Use Inter Quartile Range (IQR) or Moving Average method
                    logging.info("Less than 2 years of data - Use Moving Average or IQR Method")
                    logging.info("Default - Using IQR method for outlier detection.")
                    return detect_outliers_iqr(df_pandas)
        return detect_outliers_stl_extended(df_stl, detected_period)
    else:
        print("Duplicate date index values. Check your data.")


def detect_outliers_stl_extended(df, detected_period) -> Union[pd.DataFrame, str]:
    """
        Method for calling core outlier function to detect Seasonal-Trend Decomposition using LOESS (STL)

        Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                and last column should be a count/feature column.

        Returns:
            str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
        """

    derived_seasonal = detected_period + ((detected_period % 2) == 0)  # Ensure odd
    logging.info("Detected Period: %d", detected_period)
    logging.info("Derived Seasonal: %d", derived_seasonal)

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = STL(df_box['transformed_data'], seasonal=derived_seasonal, period=detected_period).fit()

    result_add = STL(df.iloc[:, -1], seasonal=derived_seasonal, period=detected_period).fit()

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) < np.var(result_add.resid):
        logging.info("Multiplicative model detected")
        type = 'multiplicative'
        df_outliers = generate_outliers_stl(df, type, derived_seasonal, detected_period)
    else:
        logging.info("Additive model detected")
        type = 'additive'
        df_outliers = generate_outliers_stl(df, type, derived_seasonal, detected_period)

    return_outliers = df_outliers.iloc[:, :2]
    return_outliers.reset_index(drop=True, inplace=True)
    logging.info("Completing outlier detection using STL")
    return return_outliers


def generate_outliers_stl(df, type, seasonal, period) -> pd.DataFrame:
    """
    Generate outliers in a time-series dataset through Seasonal-Trend Decomposition using LOESS (STL)

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.
         type: Additive or Multiplicative: STL model type
         Period: Period parameter should be set to the amount of times we expect
            the seasonal cycle to re-occur within a year.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """
    # Using STL to detect outliers
    logging.info("Generating outlier detection using STL")

    if type == 'additive':
        logging.info("Outlier detection using STL Additive Model")
        df_add = df.copy()
        stl = STL(df_add.iloc[:, -1], seasonal=seasonal, period=period)
        result = stl.fit()

        # Access the residual component
        residuals = result.resid

    else:
        logging.info("Outlier detection using STL Multiplicative Model")
        df_mul = df.copy()
        df_mul['count'] = df_mul.iloc[:, -1].astype('float64')

        # Apply Box-Cox transformation
        transformed_data, _ = stats.boxcox(df_mul['count'])

        df_mul['count'] = transformed_data
        stl = STL(df_mul['count'], seasonal=seasonal, period=period)
        result = stl.fit()

        # Back-transform if Box-Cox was applied
        residual_transformed = inv_boxcox(result.resid, _)

        # Access the residual component
        residuals = residual_transformed

    # Using Median Absolute Deviation (MAD) to detect outliers
    is_outlier = anomaly_mad(residuals)
    anomalies = df[is_outlier]
    logging.info("Generated outlier detection using STL")
    return anomalies


def detect_outliers_mstl(df) -> Union[pd.DataFrame, str]:
    """
    Detect outliers in a time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).
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
                derived_period = (period_weekly, period_yearly)
            case 'MS' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly
            case 'M' if length_index >= 24:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # month level time-series.")
                period_monthly = 12
                derived_period = period_monthly
            case 'Q' if length_index >= 8:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # quarter level time-series.")
                period_quarterly = 4
                period_yearly = 12
                derived_period = (period_quarterly, period_yearly)
            case 'A' if length_index >= 2:
                # logging.info("Using seasonal trend decomposition for for outlier detection in
                # annual level time-series.")
                derived_period = 1  # Annual seasonality
            case _:
                if regex.match(regex_week_check, inferred_frequency) and length_index >= 104:
                    derived_period = 52  # Week level seasonality
                else:
                    # If less than 2 years of data, Use Moving Average or Inter Quartile Range (IQR) method
                    logging.info("Less than 2 years of data - Use IQR or Moving Average Method")
                    logging.info("Default - Using IQR method for outlier detection.")
                    return detect_outliers_iqr(df_pandas)
        return detect_outliers_mstl_extended(df_mstl, derived_period)
    else:
        print("Duplicate date index values. Check your data.")


def detect_outliers_mstl_extended(df, derived_period) -> Union[pd.DataFrame, str]:
    """
     Method for calling core function to detect Seasonal-Trend Decomposition using LOESS (STL)

        Args:
            df (pd.DataFrame): A Pandas DataFrame with time-series data.
                First column must be a date column ('YYYY-MM-DD')
                 and last column should be a count/feature column.

        Returns:
                str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
     """

    logging.info("Derived Period: %s", derived_period)

    # Try both additive and multiplicative models before selecting the right one
    # Apply Box-Cox transformation for multiplicative model
    df_box = df.copy()
    df_box['count'] = df.iloc[:, -1].astype('float64')
    df_box['transformed_data'], _ = stats.boxcox(df_box['count'])
    result_mul = MSTL(df_box['transformed_data'], periods=derived_period).fit()
    result_add = MSTL(df.iloc[:, -1], periods=derived_period).fit()

    # Choose the model with lower variance in residuals
    if np.var(result_mul.resid) < np.var(result_add.resid):
        # logging.info("Multiplicative model detected")
        type = 'multiplicative'
        df_outliers = generate_outliers_mstl(df, type, derived_period)
    else:
        # logging.info("Additive model detected")
        type = 'additive'
        df_outliers = generate_outliers_mstl(df, type, derived_period)

    return_outliers = df_outliers.iloc[:, :2]
    return_outliers.reset_index(drop=True, inplace=True)
    logging.info("Completing outlier detection using MSTL")
    return return_outliers


def generate_outliers_mstl(df, type, period) -> pd.DataFrame:
    """
    Generate outliers in a time-series dataset using Multiple Seasonal-Trend decomposition using Loess (MSTL).

    Args:
        df (pd.DataFrame): A Pandas DataFrame with time-series data.
            First column must be a date column ('YYYY-MM-DD')
            and last column should be a count/feature column.
         type: Additive or Multiplicative: STL model type
         Period: Period parameter should be set to the amount of times we expect
            the seasonal cycle to re-occur within a year.

    Returns:
        str or pd.DataFrame: A message with None found or a DataFrame with detected outliers.
    """
    # Using MSTL to detect outliers
    logging.info("Generating outlier detection using MSTL")

    if type == 'additive':
        logging.info("Outlier detection using MSTL Additive Model")
        df_add = df.copy()
        mstl = MSTL(df_add.iloc[:, -1], periods=period)
        result = mstl.fit()

        # Access the residual component
        residuals = result.resid

    else:
        logging.info("Outlier detection using MSTL Multiplicative Model")
        df_mul = df.copy()
        df_mul['count'] = df_mul.iloc[:, -1].astype('float64')

        # Apply Box-Cox transformation
        transformed_data, _ = stats.boxcox(df_mul['count'])

        df_mul['count'] = transformed_data
        mstl = MSTL(df_mul['count'], periods=period)
        result = mstl.fit()

        # Back-transform if Box-Cox was applied
        residual_transformed = inv_boxcox(result.resid, _)

        # Access the residual component
        residuals = residual_transformed

    # Using Median Absolute Deviation (MAD) to detect outliers
    is_outlier = anomaly_mad(residuals)
    anomalies = df[is_outlier]
    logging.info("Generated outlier detection using MSTL")
    return anomalies


def detect_outliers_today_stl(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect the outliers detected today using STL seasonal decomposition method

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing today's outliers if detected.
        str: A message indicating no outliers were found today.
    """

    logging.info("Detecting today's outliers.")

    # Get the DataFrame of outliers from detect_outliers and select the latest row
    df_outliers = detect_outliers_stl(df)
    df_last_outlier = df_outliers.tail(1)

    # Ensure the index is a datetime object
    df_last_outlier.index = pd.to_datetime(df_last_outlier.index)

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


def detect_outliers_latest_stl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the last outliers detected using the detect_outlier_stl method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the latest detected outlier.
    """

    logging.info("Detecting the latest outliers.")

    df_outliers = detect_outliers_stl(df)
    df_latest_outlier = df_outliers.tail(1)

    logging.info("Detected the latest outlier!")

    return df_latest_outlier


def detect_outliers_today_mstl(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """
    Detect the outliers detected today using MSTL seasonal decomposition method

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing today's outliers if detected.
        str: A message indicating no outliers were found today.
    """

    logging.info("Detecting today's outliers.")

    # Get the DataFrame of outliers from detect_outliers and select the latest row
    df_outliers = detect_outliers_mstl(df)
    df_last_outlier = df_outliers.tail(1)

    # Ensure the index is a datetime object
    df_last_outlier.index = pd.to_datetime(df_last_outlier.index)

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


def detect_outliers_latest_mstl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the last outliers detected using the detect_outlier_stl method.

    Args:
         df (pd.DataFrame): A DataFrame containing the data. The first column should be the date,
                           and the last column should be the feature (count) for which outliers are detected.

    Returns:
        pd.DataFrame: A DataFrame containing the latest detected outlier.
    """

    logging.info("Detecting the latest outliers.")

    df_outliers = detect_outliers_mstl(df)
    df_latest_outlier = df_outliers.tail(1)

    logging.info("Detected the latest outlier!")
    return df_latest_outlier
