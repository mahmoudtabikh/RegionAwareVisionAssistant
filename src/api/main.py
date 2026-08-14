import onnxruntime as ort
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(category: str, file: UploadFile = File(...)):
    contents = await file.read()  # raw bytes
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = process_image(image)
    session = load_onnx_session(category)
    results = run_onnx_inference(session, image)
    return results
