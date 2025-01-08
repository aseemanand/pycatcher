from fastapi.testclient import TestClient
from src.pycatcher.api_catch import app
import pandas as pd

# Create a TestClient instance for the FastAPI app
client = TestClient(app)

def test_find_outliers_api():
    # Create a TestClient instance for the FastAPI app
    client = TestClient(app)

    # Prepare test data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10, 12, 14, 100, 15]
    }

    df = pd.DataFrame(data)
    payload = {
        "data": df.values.tolist(),  # Convert DataFrame rows to a list of lists
        "columns": df.columns.tolist()  # Get the list of column names
    }

    # Send a POST request to the API endpoint
    response = client.post("/find_outliers", json=payload)

    # Expected output
    expected_outliers = [
        {'ID': '1970-01-01T00:00:00', 'Value': 100.0, 'index': 3}
    ]

    # Check the HTTP status code
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Parse the JSON response
    response_data = response.json()

    # Validate the response format
    assert "outliers" in response_data, "Response does not contain 'outliers' key"

    # Validate the outliers detected
    assert response_data["outliers"] == expected_outliers, (
        f"Expected outliers {expected_outliers}, but got {response_data['outliers']}"
    )

def test_detect_outliers_stl_api():
    # Prepare test data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10, 12, 14, 100, 15]
    }
    df = pd.DataFrame(data)
    payload = {
        "data": df.values.tolist(),
        "columns": df.columns.tolist()
    }

    response = client.post("/detect_outliers_stl", json=payload)

    # Expected output
    expected_outliers = [
        {'ID': '1970-01-01T00:00:00', 'Value': 100.0, 'index': 3}
    ]

    assert response.status_code == 200
    response_data = response.json()
    assert "outliers" in response_data

    # Validate the outliers detected
    assert response_data["outliers"] == expected_outliers, (
        f"Expected outliers {expected_outliers}, but got {response_data['outliers']}"
    )

def test_detect_outliers_today_classic_api():
    # Prepare test data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10, 12, 14, 100, 15]
    }

    df = pd.DataFrame(data)
    payload = {
        "data": df.values.tolist(),
        "columns": df.columns.tolist()
    }

    response = client.post("/detect_outliers_today_classic", json=payload)

    # Expected output
    expected_outliers = [
        {'message': 'No Outliers Today!'}
    ]

    assert response.status_code == 200
    response_data = response.json()
    assert "outliers" in response_data

    # Validate the outliers detected
    assert response_data["outliers"] == expected_outliers, (
        f"Expected outliers {expected_outliers}, but got {response_data['outliers']}"
    )

    print("Test passed: The API detected outliers correctly.")
