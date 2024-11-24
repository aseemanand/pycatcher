from typing import List, Optional
from pathlib import Path
import pandas as pd
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash
)
from tabulate import tabulate
from . import create_app
from .catch import detect_outliers_iqr


class FileValidator:
    """Handles file validation logic."""

    def __init__(self, allowed_extensions: List[str]):
        self.allowed_extensions = allowed_extensions

    def is_allowed_file(self, filename: str) -> bool:
        """
        Check if the given filename has an allowed extension.

        Args:
            filename (str): Name of the file to check

        Returns:
            bool: True if file extension is allowed, False otherwise
        """
        return ('.' in filename and
                filename.rsplit('.', 1)[1].lower() in self.allowed_extensions)


class OutlierAnalyzer:
    """Handles outlier detection and analysis."""

    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder

    def process_file(self, file) -> tuple[Optional[str], Optional[str]]:
        """
        Process the uploaded file and detect outliers.

        Args:
            file: The uploaded file object

        Returns:
            tuple: (message, table_html) or (error_message, None)
        """
        try:
            file_path = Path(self.upload_folder) / file.filename
            file.save(str(file_path))

            df = pd.read_csv(file_path)
            df_outliers = detect_outliers_iqr(df)

            if not isinstance(df_outliers, pd.DataFrame):
                return f"File processed: {df_outliers}", None

            table_html = tabulate(df_outliers, headers='keys', tablefmt='html')
            message = (f"File '{file.filename}' uploaded successfully! "
                       f"Found outliers in data shape: {df_outliers.shape}")

            return message, table_html

        except Exception as e:
            return f"Error processing file: {str(e)}", None


def register_routes(app: Flask) -> None:
    """
    Register routes for the Flask application.

    Args:
        app: Flask application instance
    """
    file_validator = FileValidator(app.config['ALLOWED_EXTENSIONS'])
    outlier_analyzer = OutlierAnalyzer(app.config['UPLOAD_FOLDER'])

    @app.route("/")
    def index():
        return render_template('upload.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        if not file.filename:
            flash('No file selected')
            return redirect(request.url)

        if not file_validator.is_allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)

        message, table_html = outlier_analyzer.process_file(file)

        if table_html is None:
            flash(message)
            return redirect(url_for('index'))

        return render_template('result.html',
                               table_html=table_html,
                               message=message)

def main() -> None:
    """Initialize and run the Flask application."""
    app = create_app()
    register_routes(app)
    app.run(debug=True)
