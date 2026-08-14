import torch
import json
import numpy as np
import onnxruntime as ort
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode

# ===== CONFIGURATION =====
CATEGORY = "leather"  # change to "wood" for the other model
MVTECAD_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
CKPT_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"
ONNX_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{CATEGORY}/weights/onnx/model.onnx"
NUM_SAMPLES = 10  # how many test images to compare
VAL_SPLIT_RATIO = 0.2
SEED = 42
# ==========================

torch.set_float32_matmul_precision('medium')


def build_datamodule():
    datamodule = MVTecAD(
        root=MVTECAD_DIR,
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=1,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=VAL_SPLIT_RATIO,
        seed=SEED,
    )
    return datamodule


def load_ckpt_model():
    model = EfficientAd.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        imagenet_dir=IMAGENET_DIR,
        model_size="small",
    )
    return model


def get_ckpt_predictions(model, datamodule, num_samples):
    engine = Engine()
    datamodule.setup(stage="test")
    predictions = engine.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=CKPT_PATH,
    )
    return predictions[:num_samples]


def load_onnx_session():
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return session


def run_onnx_inference(session, image_tensor):
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    ort_inputs = {input_name: image_tensor.cpu().numpy().astype(np.float32)}
    ort_outputs = session.run(output_names, ort_inputs)
    return dict(zip(output_names, ort_outputs))


def compare_predictions(ckpt_predictions, onnx_session):
    comparison = []
    for p in ckpt_predictions:
        ckpt_score = float(p.pred_score.item())
        image_tensor = p.image

        onnx_outputs = run_onnx_inference(onnx_session, image_tensor)
        onnx_score = float(onnx_outputs["pred_score"].squeeze())

        comparison.append({
            "image_path": p.image_path[0],
            "ckpt_score": ckpt_score,
            "onnx_score": onnx_score,
            "abs_diff": abs(ckpt_score - onnx_score),
        })
    return comparison


def main():
    print(f"Comparing checkpoint vs ONNX for category: {CATEGORY}")
    datamodule = build_datamodule()
    ckpt_model = load_ckpt_model()
    ckpt_predictions = get_ckpt_predictions(ckpt_model, datamodule, NUM_SAMPLES)

    onnx_session = load_onnx_session()
    print("ONNX input info:", [(i.name, i.shape) for i in onnx_session.get_inputs()])
    print("ONNX output info:", [(o.name, o.shape) for o in onnx_session.get_outputs()])

    comparison = compare_predictions(ckpt_predictions, onnx_session)
    for c in comparison:
        print(c)
    max_diff = max(c["abs_diff"] for c in comparison)
    print(f"\nMax absolute difference across {len(comparison)} samples: {max_diff}")


if __name__ == "__main__":
    main()
    