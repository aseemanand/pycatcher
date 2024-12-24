from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the function directly from the diagnostics.py file
from pycatcher.diagnostics import build_iqr_plot

# Define the FastAPI app
app = FastAPI(
    title="Diagnostics Function API",
    description="API to expose all the diagnostics functions",
    version="1.0"
)


# Define the input model using Pydantic
class InputModel(BaseModel):
    data: list[list]  # List of lists representing the DataFrame data
    columns: list[str]  # Column names for the DataFrame


# Define the output model
class OutputModel(BaseModel):
    plot_image: str  # Base64-encoded image string


# API endpoint to expose the function
@app.post("/build_iqr_plot", response_model=OutputModel, summary="Build IQR plot for a given DataFrame")
async def build_iqr_plot_api(inputs: InputModel):
    try:
        # Convert input data into a pandas DataFrame
        df = pd.DataFrame(data=inputs.data, columns=inputs.columns)

        # Generate the IQR plot and save it to a BytesIO buffer
        # Variable 'ax' is required to avoid the type mismatch error in PyCharm and
        # to properly unpack the tuple returned by subplots().
        # However, leaving 'ax' gives prospector warning for unused variable. Ignore the warning
        fig, _ = plt.subplots()
        plt.figure()
        build_iqr_plot(df)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)

        # Encode the image to Base64
        plot_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return OutputModel(plot_image=plot_image_base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
