import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.pycatcher.catch import find_outliers_iqr, anomaly_mad, get_residuals, \
    sum_of_squares, get_ssacf, detect_outliers_today, detect_outliers_latest, \
    detect_outliers, decompose_and_detect, detect_outliers_iqr


# Test case for find_outliers_iqr
def test_find_outliers_iqr():
    # Create a sample DataFrame
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10, 12, 14, 100, 15]
    }

    df = pd.DataFrame(data)

    # Run the function
    outliers = find_outliers_iqr(df)
    print(outliers['Value'].iloc[0])

    # Assert that the outlier detected is the value 100
    assert not outliers.empty
    assert outliers['Value'].iloc[0] == 100


# Test case for anomaly_mad
def test_anomaly_mad():
    # Mock the model_type object with residuals
    mock_model = MagicMock()
    arr = np.array([1, 2, 3, 4, 100])
    mock_model.resid = pd.DataFrame(arr, columns=['Values'])

    # Mock df_pan with index
    df_pan = pd.DataFrame({"Value": [1, 2, 3, 4, 100]}, index=[0, 1, 2, 3, 4])

    # Run the function
    is_outlier = anomaly_mad(mock_model)
    df_pan = df_pan[is_outlier]

    # Assert that the outlier is detected
    assert not df_pan.empty
    assert df_pan['Value'].iloc[0] == 100


# Test case for get_residuals
def test_get_residuals():
    # Mock the model_type object with residuals
    mock_model = MagicMock()
    arr = np.array([1, 2, np.nan, 4, 5])
    mock_model.resid = pd.DataFrame(arr, columns=['Values'])

    # Run the function
    residuals = get_residuals(mock_model)

    # Check if NaNs are removed and residuals are correct
    expected = np.array([1, 2, 4, 5])
    np.testing.assert_array_equal(residuals, expected)


# Test case for sum_of_squares
def test_sum_of_squares():
    # Create a NumPy array
    array = np.array([1, 2, 3, 4])

    # Run the function
    result = sum_of_squares(array)

    # The expected sum of squares is 1^2 + 2^2 + 3^2 + 4^2 = 30
    assert result == 30


# Test case for get_ssacf
def test_get_ssacf():
    # Create residuals and df
    residuals = np.array([1, 2, 3, 4, 5])
    df = pd.DataFrame({"Value": [1, 2, 3, 4, 5]})

    # Run the function
    result = get_ssacf(residuals, df)

    # Test that the result is a valid number (more advanced checks can be added)
    assert isinstance(result, float)
    assert result >= 0


@pytest.fixture
def input_data_for_detect_outliers():
    """Fixture for sample input DataFrame."""
    return pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5),
        'value': [10, 20, 30, 40, 50]
    })


@patch('src.pycatcher.catch.detect_outliers')
def test_outliers_detected_today(mock_detect_outliers, input_data_for_detect_outliers):
    """Test case when outliers are detected today."""

    # Mock outliers DataFrame with today's date
    today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))

    df_outliers_today = pd.DataFrame({
        'date': [today],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers.return_value = df_outliers_today

    # Call the function with the sample input data
    result = detect_outliers_today(input_data_for_detect_outliers)

    # Assert that the result is the DataFrame with today's outliers
    pd.testing.assert_frame_equal(result, df_outliers_today)


@patch('src.pycatcher.catch.detect_outliers')
def test_no_outliers_today(mock_detect_outliers, input_data_for_detect_outliers):
    """Test case when no outliers are detected today."""

    # Mock outliers DataFrame with a past date (ensure the index is in datetime format)
    past_date = pd.Timestamp('2023-10-05')
    df_outliers_previous_day = pd.DataFrame({
        'date': [past_date],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers.return_value = df_outliers_previous_day

    # Call the function with the sample input data
    result = detect_outliers_today(input_data_for_detect_outliers)

    # Assert that the function returns "No Outliers Today!"
    assert result == "No Outliers Today!"


@patch('src.pycatcher.catch.detect_outliers')
def test_outliers_latest_detected(mock_detect_outliers, input_data_for_detect_outliers):
    """Test case when the latest outlier is detected."""

    # Mock outliers DataFrame with latest outlier
    latest_outlier_date = pd.Timestamp('2023-10-08')
    df_outliers = pd.DataFrame({
        'date': [latest_outlier_date],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers.return_value = df_outliers

    # Call the function with the sample input data
    result = detect_outliers_latest(input_data_for_detect_outliers)

    # Assert that the result is the DataFrame with the latest outlier
    pd.testing.assert_frame_equal(result, df_outliers.tail(1))


@patch('src.pycatcher.catch.detect_outliers')
def test_no_outliers_detected(mock_detect_outliers, input_data_for_detect_outliers):
    """Test case when no outliers are detected."""

    # Mock an empty outliers DataFrame (indicating no outliers found)
    df_no_outliers = pd.DataFrame({
        'date': [],
        'value': []
    }).set_index('date')

    mock_detect_outliers.return_value = df_no_outliers

    # Call the function with the sample input data
    result = detect_outliers_latest(input_data_for_detect_outliers)

    # Since no outliers are detected, the result should be an empty DataFrame
    pd.testing.assert_frame_equal(result, df_no_outliers)


@patch('src.pycatcher.catch.decompose_and_detect')
@patch('src.pycatcher.catch.detect_outliers_iqr')
def test_detect_outliers(mock_detect_outliers_iqr, mock_decompose_and_detect):
    # Create a sample dataset with more than 2 years of data
    date_range = pd.date_range(start='2020-01-01', periods=750, freq='D')
    data = pd.DataFrame({
        'date': date_range,
        'count': [i for i in range(len(date_range))]
    })

    # Test case where data has 2+ years (use seasonal decomposition)
    mock_decompose_and_detect.return_value = pd.DataFrame({'date': date_range[:10], 'count': [1] * 10})
    result = detect_outliers(data)
    mock_decompose_and_detect.assert_called_once()

    # Test case where data is less than 2 years (use IQR method)
    data_short = data.iloc[:300]
    mock_detect_outliers_iqr.return_value = pd.DataFrame({'date': date_range[:5], 'count': [1] * 5})
    result_short = detect_outliers(data_short)
    mock_detect_outliers_iqr.assert_called_once()

    # Test case where input is a non-Pandas DataFrame and needs conversion
    mock_spark_df = MagicMock()
    mock_pandas_df = MagicMock()

    # Mock the toPandas method to simulate conversion
    mock_spark_df.toPandas.return_value = mock_pandas_df
    mock_detect_outliers_iqr.reset_mock()
    mock_decompose_and_detect.reset_mock()

    # Assume mock_pandas_df has less than 2 years of data
    mock_pandas_df.index = range(300)  # Simulate 300 days of data
    mock_detect_outliers_iqr.return_value = pd.DataFrame({'date': date_range[:5], 'count': [1] * 5})

    result_conversion = detect_outliers(mock_spark_df)
    mock_spark_df.toPandas.assert_called_once()  # Check if conversion was called
    mock_detect_outliers_iqr.assert_called_once()  # Verify the IQR method was used for outlier detection


@pytest.fixture
def input_data_decompose_and_detect():
    """Fixture to provide sample time-series data."""
    np.random.seed(0)  # For reproducibility
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    values = np.random.normal(loc=50, scale=5, size=len(dates))
    # Introduce outliers
    values[100] = 100
    values[200] = 100
    df = pd.DataFrame({'date': dates, 'value': values})
    return df


def test_decompose_and_detect(input_data_decompose_and_detect):
    """Test case for the decompose_and_detect function."""
    # Call the function with the sample data
    result = decompose_and_detect(input_data_decompose_and_detect)

    # Check the type of result
    assert isinstance(result, (pd.DataFrame, str)), "Expected DataFrame or string as output"

    if isinstance(result, pd.DataFrame):
        # If the result is a DataFrame, check that it has outlier rows
        assert not result.empty, "Expected some outliers in the DataFrame"
        assert 'value' in result.columns, "Expected 'value' column in outliers DataFrame"
    else:
        # If the result is a string, verify it indicates no outliers
        assert result == "No outliers found.", "Expected message indicating no outliers found"


@pytest.fixture
def input_data_detect_outliers_iqr():
    """Fixture to provide sample data with outliers for testing."""
    np.random.seed(0)  # For reproducibility
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    values = np.random.normal(loc=50, scale=5, size=len(dates)).astype(int)
    # Introduce outliers
    values[50] = 100
    values[150] = 150
    values[300] = 200
    df = pd.DataFrame({'date': dates, 'value': values})
    return df


def test_detect_outliers_iqr(input_data_detect_outliers_iqr):
    """Test case for the detect_outliers_iqr function."""
    # Call the function with sample data
    result = detect_outliers_iqr(input_data_detect_outliers_iqr)

    # Check the type of result
    assert isinstance(result, (pd.DataFrame, str)), "Expected DataFrame or string as output"

    if isinstance(result, pd.DataFrame):
        # If the result is a DataFrame, check that it has outlier rows
        assert not result.empty, "Expected some outliers in the DataFrame"
        assert 'value' in result.columns, "Expected 'value' column in outliers DataFrame"
        # Verify specific known outlier values are detected
        assert {100, 150, 200}.issubset(result['value'].values), "Expected known outliers in the DataFrame"
    else:
        # If the result is a string, verify it indicates no outliers
        assert result == "No outliers found.", "Expected message indicating no outliers found"