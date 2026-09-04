import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi import HTTPException
from src.lib.inference import load_onnx_session, run_onnx_inference, process_image
from src.lib.regions import extract_regions
from src.llm.generate import setup_document_retrieval, call_qa_model_with_prediction
from langchain_ollama import OllamaLLM

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup
    app.state.sessions = {
        "leather": load_onnx_session("leather"),
        "wood": load_onnx_session("wood"),
    }
    app.state.thresholds = {
        "leather": 0.504636824131012, # grabbed from /home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_final_metrics.json
        "wood": 0.5001757740974426,  # grabbed from /home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_final_metrics.json
    }
    app.state.vector_store = setup_document_retrieval()
    app.state.llm = OllamaLLM(model="qwen3:8b")
    yield

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
    
    anomaly_map_np = results["anomaly_map"].squeeze()
    threshold = app.state.thresholds[category]
    regions = extract_regions(anomaly_map_np, threshold)
    pred_score = float(results["pred_score"].squeeze())
    
    return {
        "category": category,
        "pred_score": pred_score,
        "regions": regions,
    }

@app.post("/explain/")
async def explain(prediction: dict):
    vector_store = app.state.vector_store
    model = app.state.llm
    explanation = call_qa_model_with_prediction(vector_store, model, prediction=prediction, threshold=app.state.thresholds[prediction['category']])
    return {
        "explanation": explanation
    }
