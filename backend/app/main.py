from fastapi import FastAPI
from backend.app.api.recommend import router

app = FastAPI(title="SourceUP API")
app.include_router(router)
