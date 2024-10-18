import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from src.pycatcher.diagnostics import plot_seasonal, build_seasonal_plot


@pytest.fixture
def seasonal_decomposition_mock():
    # Mocking the seasonal decomposition result
    decomposition = MagicMock()
    decomposition.observed = MagicMock()
    decomposition.trend = MagicMock()
    decomposition.seasonal = MagicMock()
    decomposition.resid = MagicMock()
    return decomposition


def test_plot_seasonal(seasonal_decomposition_mock):
    # Create mock axes for plotting
    fig, axes = plt.subplots(ncols=1, nrows=4)

    # Call the function under test
    plot_seasonal(seasonal_decomposition_mock, axes, title="Test Title")

    # Check that the plots were called correctly
    seasonal_decomposition_mock.observed.plot.assert_called_once_with(ax=axes[0], legend=False)
    seasonal_decomposition_mock.trend.plot.assert_called_once_with(ax=axes[1], legend=False)
    seasonal_decomposition_mock.seasonal.plot.assert_called_once_with(ax=axes[2], legend=False)
    seasonal_decomposition_mock.resid.plot.assert_called_once_with(ax=axes[3], legend=False)
