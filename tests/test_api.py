import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api, "load_onnx_session", lambda category: f"{category}-session"
    )
    monkeypatch.setattr(api, "setup_document_retrieval", lambda: "vector-store")
    monkeypatch.setattr(api, "OllamaLLM", lambda **kwargs: "llm")

    with TestClient(api.app) as test_client:
        yield test_client


def test_predict_returns_prediction_and_regions(client, monkeypatch):
    decoded_image = np.zeros((10, 10, 3), dtype=np.uint8)
    processed_image = np.zeros((1, 3, 256, 256), dtype=np.float32)
    regions = [{"region_id": "region_1", "bbox": [1, 2, 3, 4]}]

    monkeypatch.setattr(api.cv2, "imdecode", lambda *_: decoded_image)
    monkeypatch.setattr(api, "process_image", lambda image: processed_image)

    def fake_inference(session, image):
        assert session == "leather-session"
        assert image is processed_image
        return {
            "anomaly_map": np.zeros((1, 2, 2), dtype=np.float32),
            "pred_score": np.array([0.75], dtype=np.float32),
        }

    extract_regions = lambda anomaly_map, threshold: regions
    monkeypatch.setattr(api, "run_onnx_inference", fake_inference)
    monkeypatch.setattr(api, "extract_regions", extract_regions)

    response = client.post(
        "/predict/?category=leather",
        files={"file": ("image.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "leather",
        "pred_score": pytest.approx(0.75),
        "regions": regions,
    }


def test_predict_rejects_invalid_category(client):
    response = client.post(
        "/predict/?category=metal",
        files={"file": ("image.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid category. Must be 'leather' or 'wood'."
    }


def test_explain_returns_model_explanation_and_uses_category_threshold(
    client, monkeypatch
):
    prediction = {"category": "wood", "pred_score": 0.8}
    expected_threshold = api.app.state.thresholds["wood"]
    calls = {}

    def fake_call_qa_model(vector_store, model, *, prediction, threshold):
        calls.update(
            vector_store=vector_store,
            model=model,
            prediction=prediction,
            threshold=threshold,
        )
        return "The image shows an anomaly."

    monkeypatch.setattr(api, "call_qa_model_with_prediction", fake_call_qa_model)

    response = client.post("/explain/", json=prediction)

    assert response.status_code == 200
    assert response.json() == {"explanation": "The image shows an anomaly."}
    assert calls == {
        "vector_store": "vector-store",
        "model": "llm",
        "prediction": prediction,
        "threshold": expected_threshold,
    }
