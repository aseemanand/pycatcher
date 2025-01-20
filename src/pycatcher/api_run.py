import uvicorn
from fastapi import FastAPI
from src.pycatcher.api_catch import app as app_catch
from src.pycatcher.api_diagnostics import app as app_diagnostics

parent_app = FastAPI()

# Mount the two apps
parent_app.mount("/catch", app_catch)
parent_app.mount("/diagnostics", app_diagnostics)


def main():
    uvicorn.run(parent_app, host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()