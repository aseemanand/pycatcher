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


class TestGetResiduals:
    """Test cases for get_residuals function."""

    def test_valid_model(self):
        """Test with valid model containing residuals."""
        mock_model = MagicMock()
        arr = np.array([1, 2, np.nan, 4, 5])
        mock_model.resid = pd.Series(arr)

        residuals = get_residuals(mock_model)

        np.testing.assert_array_equal(residuals, np.array([1, 2, 4, 5]))

    def test_none_model(self):
        """Test with None model."""
        with pytest.raises(DataValidationError, match="Input model cannot be None"):
            get_residuals(None)

    def test_model_without_resid(self):
        """Test with model missing resid attribute."""
        mock_model = MagicMock()
        del mock_model.resid  # Ensure no resid attribute

        with pytest.raises(DataValidationError, match="Model must have 'resid' attribute"):
            get_residuals(mock_model)

    def test_all_nan_residuals(self):
        """Test with model containing all NaN residuals."""
        mock_model = MagicMock()
        arr = np.array([np.nan, np.nan, np.nan])
        mock_model.resid = pd.Series(arr)

        with pytest.raises(ValueError, match="No valid residuals found after NaN removal"):
            get_residuals(mock_model)


class TestSumOfSquares:
    """Test cases for sum_of_squares function."""

    def test_valid_array(self):
        """Test with valid numpy array."""
        array = np.array([1, 2, 3, 4])
        result = sum_of_squares(array)
        assert result == 30  # 1^2 + 2^2 + 3^2 + 4^2 = 30

    def test_2d_array(self):
        """Test with 2D numpy array."""
        array = np.array([[1, 2], [3, 4]])
        result = sum_of_squares(array)
        assert result == 30  # Same result as 1D array

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input array cannot be None"):
            sum_of_squares(None)

    def test_empty_array(self):
        """Test with empty numpy array."""
        with pytest.raises(DataValidationError, match="Input array cannot be empty"):
            sum_of_squares(np.array([]))

    def test_non_numpy_array(self):
        """Test with non-numpy array input."""
        with pytest.raises(TypeError, match="Input must be a NumPy array"):
            sum_of_squares([1, 2, 3, 4])  # Regular list instead of numpy array


class TestGetSSACF:
    """Test cases for get_ssacf function."""

    def test_valid_residuals(self):
        """Test with valid numpy array residuals."""
        residuals = np.array([1, 2, 3, 4])
        result = get_ssacf(residuals, "test_model")
        assert isinstance(result, float)
        assert result > 0  # Sum of squares should be positive

    def test_2d_residuals(self):
        """Test with 2D numpy array residuals."""
        residuals = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):  # ACF expects 1D array
            get_ssacf(residuals, "test_model")

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input residuals cannot be None"):
            get_ssacf(None, "test_model")

    def test_empty_residuals(self):
        """Test with empty numpy array."""
        with pytest.raises(DataValidationError, match="Input residuals array cannot be empty"):
            get_ssacf(np.array([]), "test_model")

    def test_non_numpy_array(self):
        """Test with non-numpy array input."""
        with pytest.raises(TypeError, match="Residuals must be a NumPy array"):
            get_ssacf([1, 2, 3, 4], "test_model")


class TestDetectOutliersTodayClassic:
    """Test cases for detect_outliers_today_classic function."""

    @pytest.fixture
    def sample_df_with_outliers(self):
        """Fixture for sample DataFrame with outliers."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        return pd.DataFrame({
            'value': [10, 20, 1000, 30, 40]  # 1000 is an outlier
        }, index=dates)

    def test_outliers_today(self, sample_df_with_outliers, monkeypatch):
        """Test when outliers are detected today."""

        # Mock detect_outliers_classic to return known outliers
        def mock_detect_outliers(df):
            return sample_df_with_outliers.tail(1)

        monkeypatch.setattr("src.pycatcher.catch.detect_outliers_classic", mock_detect_outliers)

        result = detect_outliers_today_classic(sample_df_with_outliers)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_no_outliers_today(self, sample_df_with_outliers, monkeypatch):
        """Test when no outliers are detected today."""

        # Mock detect_outliers_classic to return outliers from previous day
        def mock_detect_outliers(df):
            return sample_df_with_outliers.iloc[:-1]

        monkeypatch.setattr("src.pycatcher.catch.detect_outliers_classic", mock_detect_outliers)

        result = detect_outliers_today_classic(sample_df_with_outliers)
        assert isinstance(result, str)
        assert result == "No Outliers Today!"

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            detect_outliers_today_classic(None)

    def test_empty_dataframe_no_rows(self):
        """Test with DataFrame having no rows."""
        empty_df = pd.DataFrame(columns=['value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            detect_outliers_today_classic(empty_df)

    def test_invalid_dataframe_format(self):
        """Test with invalid DataFrame format."""
        # Create DataFrame without DatetimeIndex
        df = pd.DataFrame({'value': [1, 2, 3]})
        with pytest.raises(DataValidationError, match="DataFrame must have a DatetimeIndex"):
            detect_outliers_today_classic(df)

    def test_missing_value_column(self):
        """Test with DataFrame missing value column."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=3)
        df = pd.DataFrame(index=dates)  # DataFrame with only DatetimeIndex
        with pytest.raises(DataValidationError, match="DataFrame must contain at least one value column"):
            detect_outliers_today_classic(df)


class TestDetectOutliersLatestClassic:
    """Test cases for detect_outliers_latest_classic function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        return pd.DataFrame({
            'value': [10, 20, 1000, 30, 40]  # 1000 is an outlier
        }, index=dates)

    def test_valid_detection(self, sample_df, monkeypatch):
        """Test with valid DataFrame containing outliers."""
        # Mock detect_outliers_classic to return known outliers
        mock_outliers = sample_df.iloc[[2]]  # Row with value 1000
        def mock_detect_outliers(df):
            return mock_outliers

        monkeypatch.setattr("src.pycatcher.catch.detect_outliers_classic", mock_detect_outliers)

        result = detect_outliers_latest_classic(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['value'] == 1000

    def test_no_outliers(self, sample_df, monkeypatch):
        """Test when no outliers are detected."""
        def mock_detect_outliers(df):
            return pd.DataFrame()

        monkeypatch.setattr("src.pycatcher.catch.detect_outliers_classic", mock_detect_outliers)

        result = detect_outliers_latest_classic(sample_df)
        assert result.empty

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            detect_outliers_latest_classic(None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            detect_outliers_latest_classic(empty_df)

    def test_invalid_index(self):
        """Test with invalid index type."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        with pytest.raises(DataValidationError, match="DataFrame must have a DatetimeIndex"):
            detect_outliers_latest_classic(df)


@pytest.fixture
def input_data_for_detect_outliers():
    """Fixture for sample input DataFrame."""
    return pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5),
        'value': [10, 20, 30, 40, 50]
    })


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