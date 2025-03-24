import uvicorn

from src.routes import create_app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app:app", port=3000, host="0.0.0.0", reload=False, access_log=False)
