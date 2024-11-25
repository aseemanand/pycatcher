import pytest
from src.pycatcher import create_app
from src.pycatcher.webapp import register_routes

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": "/tmp",
        "ALLOWED_EXTENSIONS": ["csv"],
        "SECRET_KEY": "testsecret",
    })
    register_routes(app)
    return app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test CLI runner for the app."""
    return app.test_cli_runner()
