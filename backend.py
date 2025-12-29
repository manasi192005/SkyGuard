from fastapi import FastAPI

app = FastAPI(title="SkyGuard API")

@app.get("/")
def root():
    return {"status": "SkyGuard API is running"}
