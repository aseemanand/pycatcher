from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the functions directly from diagnostics.py
from pycatcher.diagnostics import (
    build_iqr_plot,
    build_seasonal_plot_classic,
    build_seasonal_plot_stl,
    build_seasonal_plot_mstl,
    build_outliers_plot_classic,
    build_outliers_plot_mstl,
    build_outliers_plot_stl,
    build_outliers_plot_esd,
    build_outliers_plot_moving_average
)

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


# Utility function to handle plotting and encoding
def generate_plot_response(plot_function, df: pd.DataFrame):
    try:
        # Generate the plot using the provided function
        fig = plot_function(df)
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)

        # Encode the image to Base64
        plot_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return OutputModel(plot_image=plot_image_base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for IQR plot
@app.post("/build_iqr_plot", response_model=OutputModel, summary="Build IQR plot for a given DataFrame")
async def build_iqr_plot_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_iqr_plot, df)


# Endpoint for Classic Seasonal Plot
@app.post("/build_seasonal_plot_classic", response_model=OutputModel, summary="Build Classic Seasonal plot for a "
                                                                              "given DataFrame")
async def build_seasonal_plot_classic_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_seasonal_plot_classic, df)


# Endpoint for STL Seasonal Plot
@app.post("/build_seasonal_plot_stl", response_model=OutputModel, summary="Build STL Seasonal plot for a "
                                                                          "given DataFrame")
async def build_seasonal_plot_stl_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_seasonal_plot_stl, df)


# Endpoint for MSTL Seasonal Plot
@app.post("/build_seasonal_plot_mstl", response_model=OutputModel, summary="Build MSTL Seasonal plot for a "
                                                                           "given DataFrame")
async def build_seasonal_plot_mstl_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_seasonal_plot_mstl, df)


# Endpoint for Classic Outliers Plot
@app.post("/build_outliers_plot_classic", response_model=OutputModel, summary="Build Classic Outliers plot for a "
                                                                              "given DataFrame")
async def build_outliers_plot_classic_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_outliers_plot_classic, df)


# Endpoint for MSTL Outliers Plot
@app.post("/build_outliers_plot_mstl", response_model=OutputModel, summary="Build MSTL Outliers plot for a "
                                                                           "given DataFrame")
async def build_outliers_plot_mstl_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_outliers_plot_mstl, df)


# Endpoint for STL Outliers Plot
@app.post("/build_outliers_plot_stl", response_model=OutputModel, summary="Build STL Outliers plot for a "
                                                                          "given DataFrame")
async def build_outliers_plot_stl_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_outliers_plot_stl, df)


# Endpoint for ESD Outliers Plot
@app.post("/build_outliers_plot_esd", response_model=OutputModel, summary="Build ESD Outliers plot for a "
                                                                          "given DataFrame")
async def build_outliers_plot_esd_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_outliers_plot_esd, df)


# Endpoint for Moving Average Outliers Plot
@app.post("/build_outliers_plot_moving_average", response_model=OutputModel, summary="Build Moving Average "
                                                                                     "Outliers plot for a "
                                                                                     "given DataFrame")
async def build_outliers_plot_moving_average_api(inputs: InputModel):
    df = pd.DataFrame(data=inputs.data, columns=inputs.columns)
    return generate_plot_response(build_outliers_plot_moving_average, df)
