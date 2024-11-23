import os
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Welcome to Your Python Package Web UI</h1>"

if __name__ == "__main__":
    app.run(debug=True)
