from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1 import router as v1_router

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

app.include_router(v1_router, prefix="/v1")

app.mount("/docs/internal", 
          StaticFiles(directory="internal_docs/site"), 
          name="internal_docs")


instrumentator = Instrumentator().instrument(app).expose(app)
