from unittest.mock import MagicMock

import numpy as np
import pytest

from src.lib import inference


def test_process_image_converts_bgr_to_rgb_and_resizes():
    image = np.empty((10, 20, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]  # BGR

    result = inference.process_image(image)

    assert result.shape == (1, 3, 256, 256)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 100, 100], [30 / 255, 20 / 255, 10 / 255])


def test_load_onnx_session_uses_category_path_and_cpu_provider(monkeypatch):
    session = MagicMock()
    create_session = MagicMock(return_value=session)
    monkeypatch.setattr(inference, "BASE_PATH", "/tmp/models")
    monkeypatch.setattr(inference.ort, "InferenceSession", create_session)

    result = inference.load_onnx_session("leather")

    assert result is session
    create_session.assert_called_once_with(
        "/tmp/models/results/exports/leather/weights/onnx/model.onnx",
        providers=["CPUExecutionProvider"],
    )


@pytest.mark.parametrize("category", ["invalid", "", "metal"])
def test_load_onnx_session_rejects_invalid_category(category, monkeypatch):
    create_session = MagicMock()
    monkeypatch.setattr(inference.ort, "InferenceSession", create_session)

    with pytest.raises(ValueError, match="Invalid category"):
        inference.load_onnx_session(category)

    create_session.assert_not_called()


def test_run_onnx_inference_maps_output_names():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input_tensor")]
    session.get_inputs.return_value[0].name = "input_tensor"
    session.get_outputs.return_value = [
        MagicMock(name="heatmap"),
        MagicMock(name="score"),
    ]
    session.get_outputs.return_value[0].name = "heatmap"
    session.get_outputs.return_value[1].name = "score"

    heatmap = np.zeros((1, 1, 2, 2), dtype=np.float32)
    score = np.array([0.8], dtype=np.float32)
    session.run.return_value = [heatmap, score]
    image = np.zeros((1, 3, 256, 256), dtype=np.float32)

    result = inference.run_onnx_inference(session, image)

    assert result == {"heatmap": heatmap, "score": score}
    args, kwargs = session.run.call_args
    # run_onnx_inference calls session.run(output_names, ort_inputs) positionally,
    # not with keyword arguments.
    assert args[0] == ["heatmap", "score"]
    assert args[1] == {"input_tensor": image}
    assert kwargs == {}
