import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.pycatcher.catch import (TimeSeriesError, DataValidationError, check_and_convert_date, find_outliers_iqr,
    anomaly_mad, get_residuals, sum_of_squares, get_ssacf, detect_outliers_today_classic,
    detect_outliers_latest_classic, detect_outliers_classic, decompose_and_detect, detect_outliers_iqr,
    calculate_rmse, calculate_optimal_window_size, detect_outliers_moving_average)


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


class TestDetectOutliersClassic:
    """Test cases for detect_outliers_classic function."""

    @pytest.fixture
    def daily_df(self):
        """Fixture for daily data with 2+ years."""
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, len(dates))
        })

    @pytest.fixture
    def monthly_df(self):
        """Fixture for monthly data with 2+ years."""
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='MS')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, len(dates))
        })

    @pytest.fixture
    def weekly_df(self):
        """Fixture for weekly data with 2+ years."""
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='W')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, len(dates))
        })

    @pytest.fixture
    def short_df(self):
        """Fixture for short period data (less than 2 years)."""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, len(dates))
        })

    def test_daily_data_seasonal(self, daily_df, monkeypatch):
        """Test with daily data spanning more than 2 years."""
        mock_decompose = MagicMock(return_value=pd.DataFrame({'value': [1, 2, 3]}))
        monkeypatch.setattr("src.pycatcher.catch.decompose_and_detect", mock_decompose)

        result = detect_outliers_classic(daily_df)

        assert mock_decompose.called
        # The DataFrame passed to mock should have dates as index
        called_df = mock_decompose.call_args[0][0]
        assert isinstance(called_df.index, pd.DatetimeIndex)

    def test_monthly_data_seasonal(self, monthly_df, monkeypatch):
        """Test with monthly data spanning more than 2 years."""
        mock_decompose = MagicMock(return_value=pd.DataFrame({'value': [1, 2, 3]}))
        monkeypatch.setattr("src.pycatcher.catch.decompose_and_detect", mock_decompose)

        result = detect_outliers_classic(monthly_df)

        assert mock_decompose.called
        called_df = mock_decompose.call_args[0][0]
        assert isinstance(called_df.index, pd.DatetimeIndex)

        # Verify that the index is monthly by checking consecutive dates
        date_diffs = called_df.index[1:] - called_df.index[:-1]
        assert all(diff.days >= 28 and diff.days <= 31 for diff in date_diffs), "Data should be monthly"

    def test_weekly_data_seasonal(self, weekly_df, monkeypatch):
        """Test with weekly data spanning more than 2 years."""
        mock_decompose = MagicMock(return_value=pd.DataFrame({'value': [1, 2, 3]}))
        monkeypatch.setattr("src.pycatcher.catch.decompose_and_detect", mock_decompose)

        result = detect_outliers_classic(weekly_df)

        assert mock_decompose.called
        called_df = mock_decompose.call_args[0][0]
        assert isinstance(called_df.index, pd.DatetimeIndex)

    def test_short_period_iqr(self, short_df, monkeypatch):
        """Test with short period data (less than 2 years)."""
        mock_iqr = MagicMock(return_value=pd.DataFrame({'value': [1, 2]}))
        monkeypatch.setattr("src.pycatcher.catch.detect_outliers_iqr", mock_iqr)

        result = detect_outliers_classic(short_df)

        assert mock_iqr.called
        called_df = mock_iqr.call_args[0][0]
        assert isinstance(called_df.index, pd.DatetimeIndex)

    def test_non_pandas_dataframe(self):
        """Test with non-pandas DataFrame input that has toPandas method."""
        dates = pd.date_range(start='2022-01-01', periods=5)
        pandas_df = pd.DataFrame({
            'date': dates,
            'value': [1, 2, 3, 4, 5]
        })

        mock_df = MagicMock()
        mock_df.toPandas = MagicMock(return_value=pandas_df)

        result = detect_outliers_classic(mock_df)
        assert mock_df.toPandas.called

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            detect_outliers_classic(None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['date', 'value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            detect_outliers_classic(empty_df)

    def test_invalid_input_type(self):
        """Test with invalid input type (neither DataFrame nor has toPandas)."""
        with pytest.raises(TypeError, match="Input must be a DataFrame or have toPandas method"):
            detect_outliers_classic([1, 2, 3])

    def test_duplicate_dates(self):
        """Test with DataFrame containing duplicate dates."""
        df = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-01', '2022-01-01'],
            'value': [1, 2, 3]
        })

        with pytest.raises(DataValidationError):
            detect_outliers_classic(df)

    def test_invalid_date_format(self):
        """Test with invalid date format."""
        df = pd.DataFrame({
            'date': ['invalid', 'dates'],
            'value': [1, 2]
        })

        with pytest.raises(DataValidationError):
            detect_outliers_classic(df)

    def test_quarterly_data_seasonal(self):
        """Test with quarterly data spanning more than 2 years."""
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='Q')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, len(dates))
        })

        result = detect_outliers_classic(df)
        assert isinstance(result, (pd.DataFrame, str))

    @pytest.mark.parametrize("freq,periods", [
        ('D', 729),  # Just under 2 years daily
        ('B', 519),  # Just under 2 years business days
        ('MS', 23),  # Just under 2 years monthly
        ('Q', 7),  # Just under 2 years quarterly
        ('W', 103)  # Just under 2 years weekly
    ])
    def test_borderline_frequencies(self, freq, periods):
        """Test with borderline frequencies that should use IQR method."""
        dates = pd.date_range(start='2022-01-01', periods=periods, freq=freq)
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, periods)
        })

        with patch('src.pycatcher.catch.detect_outliers_iqr') as mock_iqr:
            mock_iqr.return_value = pd.DataFrame()
            result = detect_outliers_classic(df)
            assert mock_iqr.called
            called_df = mock_iqr.call_args[0][0]
            assert isinstance(called_df.index, pd.DatetimeIndex)


class TestDecomposeAndDetect:
    """Test cases for decompose_and_detect function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame with numeric time series data."""
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        # Create data with clear seasonal pattern and some outliers
        # Base level of 100 ensures all values are positive
        trend = np.linspace(0, 10, 100)
        seasonal = np.sin(np.linspace(0, 8 * np.pi, 100))
        noise = np.random.normal(0, 0.1, 100)
        values = 100 + trend + 5 * seasonal + noise
        # Add outliers that are still positive
        values[25] = 130  # High outlier
        values[75] = 70  # Low outlier but still positive
        return pd.DataFrame({'value': values}, index=dates)

    def test_successful_decomposition(self, sample_df):
        """Test successful decomposition and outlier detection."""
        result = decompose_and_detect(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should detect the outliers we inserted
        # Verify our inserted outliers are detected
        assert any(result.index.date == sample_df.index[25].date())
        assert any(result.index.date == sample_df.index[75].date())

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            decompose_and_detect(None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            decompose_and_detect(empty_df)

    def test_non_numeric_data(self):
        """Test with non-numeric data."""
        dates = pd.date_range(start='2022-01-01', periods=5)
        df = pd.DataFrame({'value': ['a', 'b', 'c', 'd', 'e']}, index=dates)
        with pytest.raises(DataValidationError, match="Last column must contain numeric values"):
            decompose_and_detect(df)


class TestDetectOutliersIQR:
    """Test cases for detect_outliers_iqr function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        return pd.DataFrame({
            'value': [10, 20, 1000, 30, 40]  # 1000 is an outlier
        }, index=dates)

    def test_valid_detection(self, sample_df):
        """Test with valid DataFrame containing outliers."""
        result = detect_outliers_iqr(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['value'] == 1000  # Should detect the obvious outlier

    def test_no_outliers(self):
        """Test when no outliers are present."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        normal_df = pd.DataFrame({
            'value': [10, 12, 11, 13, 12]  # No outliers
        }, index=dates)

        result = detect_outliers_iqr(normal_df)
        assert isinstance(result, str)
        assert result == "No outliers found."

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            detect_outliers_iqr(None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            detect_outliers_iqr(empty_df)

    def test_non_numeric_values(self):
        """Test with non-numeric values."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=3)
        string_df = pd.DataFrame({
            'value': ['a', 'b', 'c']
        }, index=dates)

        with pytest.raises(ValueError):
            detect_outliers_iqr(string_df)

    def test_single_value(self):
        """Test with DataFrame containing single value."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1)
        single_df = pd.DataFrame({
            'value': [10]
        }, index=dates)

        result = detect_outliers_iqr(single_df)
        assert isinstance(result, str)
        assert result == "No outliers found."

    def test_all_same_values(self):
        """Test with DataFrame containing identical values."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        identical_df = pd.DataFrame({
            'value': [10, 10, 10, 10, 10]
        }, index=dates)

        result = detect_outliers_iqr(identical_df)
        assert isinstance(result, str)
        assert result == "No outliers found."

    def test_with_missing_values(self):
        """Test with DataFrame containing NaN values."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5)
        nan_df = pd.DataFrame({
            'value': [10, np.nan, 1000, np.nan, 40]
        }, index=dates)

        # When there are NaN values, they are dropped during processing
        # and if the remaining values don't contain outliers, we expect
        # the "No outliers found." string
        result = detect_outliers_iqr(nan_df)
        assert isinstance(result, str)
        assert result == "No outliers found."


class TestCalculateRmse:
    """Test cases for calculate_rmse function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame with dates and values."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        values = np.sin(np.linspace(0, 10, 100)) * 100 + 500  # Generate sine wave pattern
        return pd.DataFrame({
            'date': dates,
            'value': values
        })

    @pytest.fixture
    def small_df(self):
        """Fixture for small DataFrame with few data points."""
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'value': [100, 102, 98, 103, 97]
        })

    def test_valid_calculation(self, sample_df):
        """Test RMSE calculation with valid input data."""
        result = calculate_rmse(sample_df, window_size=7)
        assert isinstance(result, (float, np.float64))  # Check for both Python float and numpy float64
        assert float(result) == pytest.approx(0.0)  # Convert to float and use pytest.approx for comparison

    def test_none_dataframe(self):
        """Test with None DataFrame."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            calculate_rmse(None, window_size=5)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            calculate_rmse(empty_df, window_size=5)

    def test_invalid_window_size_type(self, sample_df):
        """Test with non-integer window size."""
        with pytest.raises(TypeError, match="Window size must be an integer"):
            calculate_rmse(sample_df, window_size=5.5)

    def test_negative_window_size(self, sample_df):
        """Test with negative window size."""
        with pytest.raises(ValueError, match="Window size must be greater than 0"):
            calculate_rmse(sample_df, window_size=-1)

    def test_zero_window_size(self, sample_df):
        """Test with zero window size."""
        with pytest.raises(ValueError, match="Window size must be greater than 0"):
            calculate_rmse(sample_df, window_size=0)

    def test_missing_value_column(self):
        """Test with DataFrame missing value column."""
        df = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=5))
        with pytest.raises(DataValidationError, match="DataFrame must contain at least one value column"):
            calculate_rmse(df, window_size=3)


class TestCalculateOptimalWindowSize:
    """Test cases for calculate_optimal_window_size function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame with clean time series data."""
        dates = pd.date_range(start='2022-01-01', periods=100)
        # Creating a simple time series with a trend
        values = np.linspace(10, 100, 100) + np.random.normal(0, 5, 100)
        return pd.DataFrame({
            'date': dates,
            'value': values
        })

    def test_valid_calculation(self, sample_df):
        """Test with valid DataFrame containing clean time series data."""
        result = calculate_optimal_window_size(sample_df)
        assert isinstance(result, int)
        assert 2 <= result <= 20  # Window size should be within the range defined in the function

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            calculate_optimal_window_size(None)

    def test_empty_dataframe_no_rows(self):
        """Test with DataFrame having no rows."""
        empty_df = pd.DataFrame(columns=['date', 'value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            calculate_optimal_window_size(empty_df)

    def test_empty_dataframe_no_columns(self):
        """Test with DataFrame having no data and no columns."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            calculate_optimal_window_size(empty_df)

    def test_dataframe_with_rows_no_columns(self):
        """Test with DataFrame having rows but no columns."""
        # Create DataFrame with index but no columns
        df = pd.DataFrame(index=range(5))
        with pytest.raises(DataValidationError, match="DataFrame must contain at least one value column"):
            calculate_optimal_window_size(df)

    def test_small_dataset(self):
        """Test with a small but valid dataset."""
        # Create a dataset that's large enough for 5 splits but still relatively small
        dates = pd.date_range(start='2022-01-01', periods=25)
        # Creating simple linear trend with some noise
        values = np.linspace(1, 25, 25) + np.random.normal(0, 0.5, 25)
        small_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        result = calculate_optimal_window_size(small_df)
        assert isinstance(result, int)
        assert 2 <= result <= 20  # Window size should be within the defined range

    def test_minimum_size_dataset(self):
        """Test with a dataset at the minimum size that should work."""
        # TimeSeriesSplit with n_splits=5 requires at least n_splits + 2 samples
        dates = pd.date_range(start='2022-01-01', periods=7)
        values = range(1, 8)  # Simple increasing sequence
        min_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        with pytest.raises(ValueError, match="All RMSE values are NaN"):
            calculate_optimal_window_size(min_df)

    def test_constant_values(self):
        """Test with constant values where RMSE might be all zero."""
        dates = pd.date_range(start='2022-01-01', periods=50)
        constant_df = pd.DataFrame({
            'date': dates,
            'value': [10] * 50
        })
        result = calculate_optimal_window_size(constant_df)
        assert isinstance(result, int)
        assert 2 <= result <= 20

    @patch('src.pycatcher.catch.calculate_rmse')
    def test_all_nan_rmse_values(self, mock_calculate_rmse):
        """Test handling of all NaN RMSE values."""
        mock_calculate_rmse.return_value = np.nan
        dates = pd.date_range(start='2022-01-01', periods=50)
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(0, 1, 50)
        })
        with pytest.raises(ValueError, match="All RMSE values are NaN"):
            calculate_optimal_window_size(df)

    def test_non_numeric_values(self):
        """Test with non-numeric values in the value column."""
        df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=10),
            'value': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        })
        with pytest.raises(Exception):  # Should raise an error when trying to calculate RMSE
            calculate_optimal_window_size(df)

    def test_highly_seasonal_data(self):
        """Test with highly seasonal data."""
        dates = pd.date_range(start='2022-01-01', periods=100)
        # Create seasonal pattern with period of 7
        seasonal = np.sin(np.linspace(0, 10 * np.pi, 100)) * 10
        values = seasonal + np.random.normal(0, 1, 100)
        seasonal_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        result = calculate_optimal_window_size(seasonal_df)
        assert isinstance(result, int)
        assert 2 <= result <= 20


class TestDetectOutliersMovingAverage:
    """Test cases for detect_outliers_moving_average function."""

    @pytest.fixture
    def sample_df(self):
        """Fixture for sample DataFrame with time series data."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': [10, 12, 11, 13, 100, 11, 12, 13, 11, 12]  # 100 is an outlier
        })

    def test_valid_detection(self, sample_df, monkeypatch):
        """Test with valid DataFrame containing outliers."""

        # Mock calculate_optimal_window_size to return a fixed window size
        def mock_optimal_window(df):
            return 3

        # Mock anomaly_zscore to return known z-scores
        def mock_zscore(series):
            return pd.Series([0, 0, 0, 0, 5, 0, 0, 0, 0, 0])  # High z-score for outlier

        monkeypatch.setattr("src.pycatcher.catch.calculate_optimal_window_size", mock_optimal_window)
        monkeypatch.setattr("src.pycatcher.catch.anomaly_zscore", mock_zscore)

        result = detect_outliers_moving_average(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['value'] == 100

    def test_none_input(self):
        """Test with None input."""
        with pytest.raises(DataValidationError, match="Input DataFrame cannot be None"):
            detect_outliers_moving_average(None)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['date', 'value'])
        with pytest.raises(DataValidationError, match="Input DataFrame cannot have zero rows"):
            detect_outliers_moving_average(empty_df)

    def test_optimal_window_calculation_error(self, sample_df, monkeypatch):
        """Test handling of errors from optimal window size calculation."""

        def mock_optimal_window(df):
            raise ValueError("Error calculating optimal window size")

        monkeypatch.setattr("src.pycatcher.catch.calculate_optimal_window_size", mock_optimal_window)

        with pytest.raises(ValueError, match="Error calculating optimal window size"):
            detect_outliers_moving_average(sample_df)
