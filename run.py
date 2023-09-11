import uvicorn

if __name__ == "__main__":
    """
    Main entry point for running the FastAPI application.

    This script launches the FastAPI application using Uvicorn web server.

    Usage:
        - Run this script to start the FastAPI application.
    """
    uvicorn.run("app.app:app",
                host="0.0.0.0",
                port=8000)
