import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.outlier_detection_nke.anomaly_detection import find_outliers_iqr, anomaly_mad, get_residuals, \
    sum_of_squares, get_ssacf, get_outliers_today


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
    outliers = anomaly_mad(mock_model, df_pan)

    # Assert that the outlier is detected
    assert not outliers.empty
    assert outliers['Value'].iloc[0] == 100


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


# Test case for get_outliers_today
def test_get_outliers_today():
    # Mock the model_type and anomaly_mad function
    mock_model = MagicMock()

    mock_outliers = pd.DataFrame({
        "Value": [100],
        "Date": pd.to_datetime([pd.Timestamp.now().strftime('%Y-%m-%d')])
    })

    mock_outliers.set_index('Date', inplace=True)

    # Patch the anomaly_mad function to return the mock outliers
    with patch('src.outlier_detection_nke.anomaly_detection.anomaly_mad', return_value=mock_outliers):
        result = get_outliers_today(mock_model)

    # Assert that the outlier is detected today
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


# Test case for get_outliers_today when no outliers are present
def test_get_outliers_today_no_outliers():
    # Mock the model_type and anomaly_mad function
    mock_model = MagicMock()

    # Mock df without today's date
    mock_outliers = pd.DataFrame({
        "Value": [100],
        "Date": pd.to_datetime(['2022-01-01'])
    })

    mock_outliers.set_index('Date', inplace=True)

    # Patch the anomaly_mad function to return the mock outliers
    with patch('src.outlier_detection_nke.anomaly_detection.anomaly_mad', return_value=mock_outliers):
        result = get_outliers_today(mock_model)

    # Assert that no outliers are detected today
    assert result == "No Outliers Today!"
