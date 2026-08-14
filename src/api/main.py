import onnxruntime as ort
import numpy as np
import cv2
import torch
from torchvision.transforms.v2 import Resize
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi import HTTPException


def load_onnx_session(category):
    if category not in ["leather", "wood"]:
        raise ValueError("Invalid category. Must be 'leather' or 'wood'.")
    onnx_path = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{category}/weights/onnx/model.onnx"
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    return session


def run_onnx_inference(session, image):
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    ort_inputs = {
        input_name: image
    }

    ort_outputs = session.run(output_names, ort_inputs)

    return dict(zip(output_names, ort_outputs))

def process_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array (HWC) to torch tensor (BCHW) with batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Scale to [0, 1]
    image_tensor = image_tensor / 255.0
    
    # Apply resize with antialias (matches anomalib's preprocessing)
    resize_transform = Resize((256, 256), antialias=True)
    image_tensor = resize_transform(image_tensor)
    
    # Convert back to numpy for ONNX Runtime (BCHW format, [0, 1] range)
    image = image_tensor.numpy()
    
    return image


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
