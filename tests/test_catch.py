import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.pycatcher.catch import (TimeSeriesError, DataValidationError, check_and_convert_date, find_outliers_iqr,
    anomaly_mad, get_residuals, sum_of_squares, get_ssacf, detect_outliers_today_classic,
    detect_outliers_latest_classic, detect_outliers_classic, decompose_and_detect, detect_outliers_iqr)


@pytest.fixture
def sample_df():
    """Fixture for sample DataFrame with dates and values."""
    return pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5),
        'value': [10, 20, 30, 40, 50]
    })


class TestCheckAndConvertDate:
    def test_valid_datetime_column(self, sample_df):
        """Test with already datetime column."""
        result = check_and_convert_date(sample_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_string_date_column(self):
        """Test with string dates that need conversion."""
        df = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-02'],
            'value': [1, 2]
        })
        result = check_and_convert_date(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_invalid_date_format(self):
        """Test with invalid date format."""
        df = pd.DataFrame({
            'date': ['invalid', 'dates'],
            'value': [1, 2]
        })
        with pytest.raises(DataValidationError):
            check_and_convert_date(df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError):
            check_and_convert_date(df)


class TestFindOutliersIQR:
    def test_normal_distribution(self):
        """Test with normally distributed data."""
        np.random.seed(42)
        normal_data = np.random.normal(loc=0, scale=1, size=1000)
        df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=1000),
            'value': normal_data
        })
        outliers = find_outliers_iqr(df)

        # Expected outliers in normal distribution: ~0.7%
        assert 0.001 <= len(outliers) / len(df) <= 0.02

    def test_with_known_outliers(self):
        """Test with dataset containing known outliers."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=5),
            'value': [1, 2, 3, 100, 4]  # 100 is a clear outlier
        })
        outliers = find_outliers_iqr(df)
        assert len(outliers) == 1
        assert outliers.iloc[0]['value'] == 100

    def test_invalid_data_type(self):
        """Test with non-numeric data."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=3),
            'value': ['a', 'b', 'c']
        })
        with pytest.raises(DataValidationError):
            find_outliers_iqr(df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError):
            find_outliers_iqr(df)


class TestAnomalyMAD:
    def test_normal_distribution(self):
        """Test with normally distributed data."""
        np.random.seed(42)
        normal_data = np.random.normal(loc=0, scale=1, size=1000)
        outliers = anomaly_mad(normal_data)
        # Expected outliers should be reasonable for normal distribution
        assert 0.001 <= np.mean(outliers) <= 0.1

    def test_with_known_outliers(self):
        """Test with known outliers."""
        data = np.array([1, 2, 3, 100, 4])  # 100 is an outlier
        outliers = anomaly_mad(data)
        assert outliers[3]  # The outlier should be detected

    def test_empty_input(self):
        """Test with empty input."""
        with pytest.raises(DataValidationError):
            anomaly_mad(np.array([]))

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError):
            anomaly_mad(None)


# Test case for anomaly_mad
def test_anomaly_mad():
    # Mock the model_type object with residuals
    mock_model = MagicMock()
    arr = np.array([1, 2, 3, 4, 100])
    mock_model.residuals = pd.DataFrame(arr, columns=['Values'])

    # Mock df_pan with index
    df_pan = pd.DataFrame({"Value": [1, 2, 3, 4, 100]}, index=[0, 1, 2, 3, 4])

    # Run the function
    is_outlier = anomaly_mad(mock_model.residuals)
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
    result = get_ssacf(residuals,type="none")

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


@patch('src.pycatcher.catch.detect_outliers_classic')
def test_outliers_detected_today(mock_detect_outliers_classic, input_data_for_detect_outliers):
    """Test case when outliers are detected today."""

    # Mock outliers DataFrame with today's date
    today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))

    df_outliers_today = pd.DataFrame({
        'date': [today],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers_classic.return_value = df_outliers_today

    # Call the function with the sample input data
    result = detect_outliers_today_classic(input_data_for_detect_outliers)

    # Assert that the result is the DataFrame with today's outliers
    pd.testing.assert_frame_equal(result, df_outliers_today)


@patch('src.pycatcher.catch.detect_outliers_classic')
def test_no_outliers_today(mock_detect_outliers_classic, input_data_for_detect_outliers):
    """Test case when no outliers are detected today."""

    # Mock outliers DataFrame with a past date (ensure the index is in datetime format)
    past_date = pd.Timestamp('2023-10-05')
    df_outliers_previous_day = pd.DataFrame({
        'date': [past_date],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers_classic.return_value = df_outliers_previous_day

    # Call the function with the sample input data
    result = detect_outliers_today_classic(input_data_for_detect_outliers)

    # Assert that the function returns "No Outliers Today!"
    assert result == "No Outliers Today!"


@patch('src.pycatcher.catch.detect_outliers_classic')
def test_outliers_latest_detected(mock_detect_outliers_classic, input_data_for_detect_outliers):
    """Test case when the latest outlier is detected."""

    # Mock outliers DataFrame with latest outlier
    latest_outlier_date = pd.Timestamp('2023-10-08')
    df_outliers = pd.DataFrame({
        'date': [latest_outlier_date],
        'value': [100]
    }).set_index('date')

    mock_detect_outliers_classic.return_value = df_outliers

    # Call the function with the sample input data
    result = detect_outliers_latest_classic(input_data_for_detect_outliers)

    # Assert that the result is the DataFrame with the latest outlier
    pd.testing.assert_frame_equal(result, df_outliers.tail(1))


@patch('src.pycatcher.catch.detect_outliers_classic')
def test_no_outliers_detected(mock_detect_outliers_classic, input_data_for_detect_outliers):
    """Test case when no outliers are detected."""

    # Mock an empty outliers DataFrame (indicating no outliers found)
    df_no_outliers = pd.DataFrame({
        'date': [],
        'value': []
    }).set_index('date')

    mock_detect_outliers_classic.return_value = df_no_outliers

    # Call the function with the sample input data
    result = detect_outliers_latest_classic(input_data_for_detect_outliers)

    # Since no outliers are detected, the result should be an empty DataFrame
    pd.testing.assert_frame_equal(result, df_no_outliers)


@patch('src.pycatcher.catch.decompose_and_detect')
@patch('src.pycatcher.catch.detect_outliers_iqr')
def test_detect_outliers(mock_detect_outliers_iqr, mock_decompose_and_detect):
    # Test Case 1: Daily frequency ('D') with 2+ years (use seasonal decomposition)
    date_range_daily = pd.date_range(start='2020-01-01', periods=750, freq='D')
    df_daily = pd.DataFrame({'date': date_range_daily, 'count': range(750)})
    mock_decompose_and_detect.return_value = pd.DataFrame({'date': date_range_daily[:10], 'count': [1] * 10})

    result_daily = detect_outliers_classic(df_daily)
    mock_decompose_and_detect.assert_called_once()
    assert isinstance(result_daily, pd.DataFrame)
    assert result_daily.equals(mock_decompose_and_detect.return_value)

    # Test Case 2: Business day frequency ('B') with more than 2 years (use seasonal decomposition)
    date_range_business = pd.date_range(start='2020-01-01', periods=520, freq='B')
    df_business = pd.DataFrame({'date': date_range_business, 'count': range(520)})
    mock_decompose_and_detect.reset_mock()  # Reset mock for separate assertions
    mock_decompose_and_detect.return_value = pd.DataFrame({'date': date_range_business[:10], 'count': [1] * 10})

    result_business = detect_outliers_classic(df_business)
    mock_decompose_and_detect.assert_called_once()
    assert isinstance(result_business, pd.DataFrame)
    assert result_business.equals(mock_decompose_and_detect.return_value)

    # Test Case 3: Weekly frequency ('W') with at least 104 entries (use seasonal decomposition)
    date_range_weekly = pd.date_range(start='2020-01-01', periods=104, freq='W')
    df_weekly = pd.DataFrame({'date': date_range_weekly, 'count': range(104)})
    mock_decompose_and_detect.reset_mock()
    mock_decompose_and_detect.return_value = pd.DataFrame({'date': date_range_weekly[:10], 'count': [1] * 10})

    result_weekly = detect_outliers_classic(df_weekly)
    mock_decompose_and_detect.assert_called_once()
    assert isinstance(result_weekly, pd.DataFrame)
    assert result_weekly.equals(mock_decompose_and_detect.return_value)

    # Test Case 4: Data with insufficient entries for seasonal decomposition (use IQR)
    date_range_short = pd.date_range(start='2021-01-01', periods=300, freq='D')
    df_short = pd.DataFrame({'date': date_range_short, 'count': range(300)})
    mock_detect_outliers_iqr.return_value = pd.DataFrame({'date': date_range_short[:5], 'count': [1] * 5})

    result_short = detect_outliers_classic(df_short)
    mock_detect_outliers_iqr.assert_called_once()
    assert isinstance(result_short, pd.DataFrame)
    assert result_short.equals(mock_detect_outliers_iqr.return_value)

    # Test Case 5: Non-Pandas DataFrame input requiring conversion to Pandas DataFrame
    mock_spark_df = MagicMock()  # Simulate a Spark DataFrame
    mock_pandas_df = pd.DataFrame({'date': date_range_short, 'count': range(300)})
    mock_spark_df.toPandas.return_value = mock_pandas_df
    mock_detect_outliers_iqr.reset_mock()

    result_conversion = detect_outliers_classic(mock_spark_df)
    mock_spark_df.toPandas.assert_called_once()
    mock_detect_outliers_iqr.assert_called_once()
    assert isinstance(result_conversion, pd.DataFrame)
    assert result_conversion.equals(mock_detect_outliers_iqr.return_value)


@pytest.fixture
def input_data_decompose_and_detect():
    """Fixture to provide sample time-series data."""
    np.random.seed(0)  # For reproducibility
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    dates.freq = 'D'  # Explicitly set the frequency
    values = np.random.normal(loc=50, scale=5, size=len(dates))
    # Introduce outliers
    values[100] = 100
    values[200] = 100
    df = pd.DataFrame({'date': dates, 'value': values})
    return df.set_index('date')


@patch('src.pycatcher.catch.get_residuals')
@patch('src.pycatcher.catch.get_ssacf')
@patch('src.pycatcher.catch.anomaly_mad')
def test_decompose_and_detect(mock_anomaly_mad, mock_get_ssacf, mock_get_residuals, input_data_decompose_and_detect):
    """Test case for the decompose_and_detect function."""

    # Mock residuals
    mock_residuals = MagicMock(spec=pd.Series)
    mock_get_residuals.return_value = mock_residuals

    # Mock SSACF values
    mock_get_ssacf.side_effect = [0.5, 0.8]  # Additive model preferred over multiplicative

    # Mock outlier detection boolean series
    mock_anomaly_mad.return_value = input_data_decompose_and_detect['value'] > 90  # Marking outliers

    # Call the function with the sample data
    result = decompose_and_detect(input_data_decompose_and_detect)

    # Check that the residuals and SSACF were calculated for both models
    assert mock_get_residuals.call_count == 2, "Expected residuals to be calculated twice (additive and multiplicative)"
    assert mock_get_ssacf.call_count == 2, "Expected SSACF to be calculated twice (additive and multiplicative)"

    # Check that anomaly detection was called with the correct model (additive, based on SSACF comparison)
    mock_anomaly_mad.assert_called_once()

    # Check the result type and contents
    assert isinstance(result, pd.DataFrame), "Expected DataFrame as output"
    assert not result.empty, "Expected some outliers in the DataFrame"
    assert 'value' in result.columns, "Expected 'value' column in outliers DataFrame"

    # Validate the outliers detected match our expectation
    expected_outliers = input_data_decompose_and_detect[input_data_decompose_and_detect['value'] > 90]
    pd.testing.assert_frame_equal(result, expected_outliers)


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