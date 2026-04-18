"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .dependencies import init_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


app = FastAPI(
    title="finFindR",
    description="Automated dorsal fin identification API — Python port of haimeh/finFindR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the web UI at /
_static_dir = __file__.replace("main.py", "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(f"{_static_dir}/index.html")
