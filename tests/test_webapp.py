import pytest
from pathlib import Path


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary sample CSV file for testing."""
    file_path = tmp_path / "sample.csv"
    file_path.write_text("col1,col2\n2024-11-01,2\n2024-11-02,2\n2024-11-03,10")
    return file_path


@pytest.fixture
def invalid_file(tmp_path):
    """Create a temporary invalid file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is not a CSV file")
    return file_path


def test_index(client):
    """Test the index route."""
    response = client.get("/")
    assert response.status_code == 200


def test_upload_no_file(client):
    """Test upload with no file selected."""
    response = client.post("/upload")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert data["message"] == "No file selected"


def test_upload_invalid_file(client, invalid_file):
    """Test upload with an invalid file type."""
    with open(invalid_file, "rb") as f:
        response = client.post("/upload", data={"file": (f, "sample.txt")})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert data["message"] == "Invalid file type. Please upload a CSV file."


def test_upload_valid_csv(client, sample_csv_file):
    """Test upload with a valid CSV file."""
    with open(sample_csv_file, "rb") as f:
        response = client.post(
            "/upload",
            data={"file": (f, "sample.csv")},
            headers={"X-Requested-With": "XMLHttpRequest"}  # Set AJAX header
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None, "Expected a JSON response but got None"
    assert data["success"] is True


def test_upload_and_analyze(client, sample_csv_file):
    """Test outlier analysis on a valid CSV file."""
    with open(sample_csv_file, "rb") as f:
        response = client.post(
            "/upload",
            data={"file": (f, "sample.csv")},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None, "Expected a JSON response but got None"
    assert data["success"] is True, f"Error message: {data.get('message', 'No message')}"


def test_file_saved_correctly(client, sample_csv_file):
    """Test if the uploaded file is saved in the specified folder."""
    upload_folder = client.application.config["UPLOAD_FOLDER"]

    with open(sample_csv_file, "rb") as f:
        client.post("/upload", data={"file": (f, "sample.csv")})

    # Check if the file was saved
    saved_file_path = Path(upload_folder) / "sample.csv"
    assert saved_file_path.exists()
