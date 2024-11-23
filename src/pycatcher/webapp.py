import os
import pandas as pd
from tabulate import tabulate
from .catch import detect_outliers_iqr
from flask import (Flask, request, render_template,
                   redirect, url_for, flash)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    # return "<h1>Welcome to Your Python Package Web UI</h1>"
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the CSV file into a Pandas DataFrame
        try:
            df = pd.read_csv(file_path)
            flash(f'File uploaded and read successfully! DataFrame shape: {df.shape}')

            # Checking if this is an outlier by using IQR
            df_iqr_outliers = detect_outliers_iqr(df)

            if not isinstance(df_iqr_outliers, pd.DataFrame):
                res = "<h1>File uploaded and read successfully! " + df_iqr_outliers + "!!</h1>"

                # Return the response
                return res
            else:
                flash(f'Finding Outliers with IQR! DataFrame shape: {df_iqr_outliers.shape}')
                # Custom message to display with the table
                message = f"File '{file.filename}' uploaded successfully! Here are the outliers!!"

                res = tabulate(df_iqr_outliers, headers='keys', tablefmt='html')

                # Render the table and message in the template
                return render_template('result.html', table_html=res, message=message)

        except Exception as e:
            flash(f'Error reading file: {e}')
            return redirect(url_for('home'))

    else:
        flash('Invalid file type. Please upload a CSV file to identify the outliers.')
        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
