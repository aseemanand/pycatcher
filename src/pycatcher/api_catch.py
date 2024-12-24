from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.pycatcher.catch import find_outliers_iqr

# Define the FastAPI app
app = FastAPI(
    title="Outliers Detection API",
    description="API to expose the `find_outliers_iqr` function for detecting outliers in a dataset.",
    version="1.0"
)


# Define the input model using Pydantic
class InputModel(BaseModel):
    data: List[List[float]]  # List of rows, each row is a list of values
    columns: List[str]       # Column names for the DataFrame


# Define the output model
class OutputModel(BaseModel):
    outliers: List[dict]  # A list of outliers as dictionaries


@app.post("/find_outliers", response_model=OutputModel, summary="Detect outliers using IQR")
async def find_outliers_api(inputs: InputModel):
    try:
        # Convert input data into a pandas DataFrame
        df = pd.DataFrame(data=inputs.data, columns=inputs.columns)

        # Ensure the first column is in datetime format (if applicable)
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error converting first column to datetime: {e}")

        # Call the `find_outliers_iqr` function
        if not callable(find_outliers_iqr):
            raise HTTPException(status_code=500, detail="`find_outliers_iqr` is not defined or not callable")

        # Call the `find_outliers_iqr` function
        outliers_df = find_outliers_iqr(df)

        # Convert the outliers DataFrame to a list of dictionaries
        outliers_list = outliers_df.reset_index().to_dict(orient="records")

        return OutputModel(outliers=outliers_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))