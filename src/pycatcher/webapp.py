import pandas as pd
from pathlib import Path
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    jsonify,
    flash
)
from . import create_app
from .catch import detect_outliers_iqr
from typing import List, Dict, Any


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

    def process_file(self, file) -> Dict[str, Any]:
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
                return {
                    'success': True,
                    'message': f"Output: {df_outliers}",
                    'data': None
                }

            # Calculate summary statistics
            total_rows = len(df)
            outlier_rows = len(df_outliers)
            outlier_percentage = (outlier_rows / total_rows) * 100

            # Format table with styling
            styled_df = df_outliers.style \
                .set_properties(**{'text-align': 'center'}) \
                .format(precision=2) \
                .hide(axis='index') \
                .to_html()

            return {
                'success': True,
                'message': 'Analysis completed successfully',
                'data': {
                    'table': styled_df,
                    'summary': {
                        'total_rows': total_rows,
                        'outlier_rows': outlier_rows,
                        'outlier_percentage': round(outlier_percentage, 2),
                        'columns_analyzed': list(df.columns)
                    }
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing file: {str(e)}",
                'data': None
            }


def register_routes(app: Flask) -> None:
    """Register routes for the Flask application."""
    file_validator = FileValidator(app.config['ALLOWED_EXTENSIONS'])
    outlier_analyzer = OutlierAnalyzer(app.config['UPLOAD_FOLDER'])

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'data': {}
            })

        file = request.files['file']

        if not file.filename:
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'data': {}
            })

        if not file_validator.is_allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Please upload a CSV file.'
            })

        result = outlier_analyzer.process_file(file)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or app.config["TESTING"]:
            return jsonify(result)

        if not result['success']:
            flash(result['message'], 'error')
            return redirect(url_for('index'))

        return render_template('result.html',
                               result=result['data'],
                               message=result['message'])


def main() -> None:
    """Initialize and run the Flask application."""
    app = create_app()
    register_routes(app)
    app.run(debug=True)