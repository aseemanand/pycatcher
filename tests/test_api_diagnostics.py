import pytest
from fastapi.testclient import TestClient
from src.pycatcher.api_diagnostics import app

# Initialize the TestClient with the FastAPI app
client = TestClient(app)

# Sample data to use in tests (ensuring sufficient rows)
sample_data = {
    "data": [
        ["2023-01-01", 100],
        ["2023-01-02", 200],
        ["2023-01-03", 300],
        ["2023-01-04", 400],
        ["2023-01-05", 500],
        ["2023-01-06", 600],
        ["2023-01-07", 700]
    ],
    "columns": ["date", "value"]
}

# Helper function to test endpoints with detailed debugging
def run_endpoint_test(endpoint: str):
    response = client.post(endpoint, json=sample_data)
    assert response.status_code == 200, (
        f"{endpoint} failed with status code {response.status_code} and response: {response.json()}"
    )
    assert "plot_image" in response.json(), (
        f"{endpoint} did not return 'plot_image'. Response: {response.json()}"
    )
    print(f"{endpoint} passed.")

# Test functions for each endpoint
def test_build_iqr_plot():
    run_endpoint_test("/build_iqr_plot")

def test_build_seasonal_plot_classic():
    run_endpoint_test("/build_seasonal_plot_classic")

def test_build_seasonal_plot_stl():
    run_endpoint_test("/build_seasonal_plot_stl")

def test_build_seasonal_plot_mstl():
    run_endpoint_test("/build_seasonal_plot_mstl")

#def test_build_outliers_plot_classic():
#    run_endpoint_test("/build_outliers_plot_classic")

def test_build_outliers_plot_mstl():
    run_endpoint_test("/build_outliers_plot_mstl")

def test_build_outliers_plot_stl():
    run_endpoint_test("/build_outliers_plot_stl")

def test_build_outliers_plot_esd():
    run_endpoint_test("/build_outliers_plot_esd")

#def test_build_outliers_plot_moving_average():
#    run_endpoint_test("/build_outliers_plot_moving_average")

if __name__ == "__main__":
    pytest.main([__file__])
