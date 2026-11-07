"""Fruit Vision - FastAPI server.

Run from the project root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predictor import FruitPredictor, InvalidImageError

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB is plenty for a phone photo downscaled by the browser
ALLOWED_TYPES = {"image/jpeg", "image/png"}  # JPG, JPEG and PNG


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: the model is loaded a single time,
    # not on every request (loading takes seconds, predicting takes milliseconds).
    app.state.predictor = FruitPredictor()
    yield


app = FastAPI(title="Fruit Vision", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health(request: Request):
    predictor: FruitPredictor = request.app.state.predictor
    return {"status": "ok", "model_loaded": True, "classes": predictor.class_names}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type. Please upload a JPG or PNG image.")

    # Read at most MAX+1 bytes so a huge upload can never fill the server's memory.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    predictor: FruitPredictor = request.app.state.predictor
    try:
        return predictor.predict(data)
    except InvalidImageError as exc:
        # The Content-Type header can lie; the real decode in Pillow is the final check.
        raise HTTPException(400, str(exc)) from exc
