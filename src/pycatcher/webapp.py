import logging
import io
import base64
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    jsonify,
    flash
)
import matplotlib.pyplot as plt
import matplotlib
from . import create_app
from .catch import detect_outliers_classic, detect_outliers_moving_average, detect_outliers_stl
from .diagnostics import build_iqr_plot
matplotlib.use('Agg')  # Use a non-interactive backend

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    def process_file(self, file, method: str = 'comprehensive') -> Dict[str, Any]:
        """
        Process the uploaded file and detect outliers.

        Args:
            file: The uploaded file object
            method: Outlier detection method to use

        Returns:
            dict: Results of outlier analysis
        """
        try:
            file_path = Path(self.upload_folder) / file.filename
            file.save(str(file_path))

            df = pd.read_csv(file_path)
            plot_base64 = None

            # Select outlier detection method
            if method == 'comprehensive':
                df_outliers = detect_outliers_classic(df)

                # Generate the plot
                fig = build_iqr_plot(df)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight', transparent=False)
                img.seek(0)
                plt.close(fig)

                # Convert the plot to a base64 string for embedding in HTML
                plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            elif method == 'stl':
                df_outliers = detect_outliers_stl(df)
            elif method == 'moving_average':
                df_outliers = detect_outliers_moving_average(df)
            else:
                raise ValueError("Invalid analysis method selected.")

            if not isinstance(df_outliers, pd.DataFrame):
                return {
                    'success': True,
                    'message': 'Analysis completed successfully',
                    'data': {
                        'table': 'No outliers found',
                        'summary': {
                            'total_rows': len(df),
                            'outlier_rows': 0,
                            'outlier_percentage': 0,
                            'columns_analyzed': list(df.columns)
                        },
                        'plot': None
                    }
                }

            # Calculate summary statistics
            total_rows = len(df)
            outlier_rows = len(df_outliers)
            outlier_percentage = (outlier_rows / total_rows) * 100

            # Resetting the index for pretty formatting
            df_outliers.reset_index(inplace=True)

            # Format table with styling
            styled_df = df_outliers.style \
                .set_table_styles([
                    # Header row style
                    {'selector': 'thead th',
                     'props': [('background-color', '#343a40'), ('color', 'white'), ('font-weight', 'bold')]},
                    # Hover effect for rows
                    {'selector': 'tr:hover', 'props': [('background-color', '#f1f1f1')]}
                ]) \
                .set_properties(**{
                    'border': '1px solid #ddd',
                    'padding': '8px',
                    'text-align': 'center'
                }) \
                .hide(axis='index') \
                .format(precision=2) \
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
                    },
                    'plot': plot_base64  # Include the plot in base64 format
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
        method = request.form.get('method', 'comprehensive')

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

        result = outlier_analyzer.process_file(file, method)

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
