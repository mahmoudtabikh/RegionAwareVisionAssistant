import onnxruntime as ort
import numpy as np
import cv2
import torch
from torchvision.transforms.v2 import Resize
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi import HTTPException
from lib.inference import load_onnx_session, run_onnx_inference, process_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup
    app.state.sessions = {
        "leather": load_onnx_session("leather"),
        "wood": load_onnx_session("wood"),
    }
    yield
    # (anything after yield runs once at shutdown — not needed here)

app = FastAPI(lifespan=lifespan)

@app.post("/predict/")
async def predict(category: str, file: UploadFile = File(...)):
    session = app.state.sessions.get(category)
    if session is None:
        raise HTTPException(status_code=400, detail="Invalid category. Must be 'leather' or 'wood'.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = process_image(image)
    results = run_onnx_inference(session, image)
    results = {k: v.tolist() for k, v in results.items()}
    return results
