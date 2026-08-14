import torch
import json
import numpy as np
from lib.regions import extract_regions
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode

# ===== CONFIGURATION =====
CATEGORY = "leather"  # change to "wood" for the other model
THRESHOLD = 0.5046368241  # leather's threshold — update when switching to wood
MVTECAD_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
MODEL_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"
OUTPUT_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/{CATEGORY}_test_full.json"
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


def load_model():
    model = EfficientAd.load_from_checkpoint(
        checkpoint_path=MODEL_PATH,
        imagenet_dir=IMAGENET_DIR,
        model_size="small",
    )
    return model


def run_predictions(model, datamodule):
    engine = Engine()
    datamodule.setup(stage="test")

    predictions = engine.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        ckpt_path=MODEL_PATH,
    )
    return predictions


def build_full_results(predictions, threshold):
    full_results = []
    for p in predictions:
        anomaly_map_np = p.anomaly_map.cpu().numpy().squeeze()
        regions = extract_regions(anomaly_map_np, threshold)

        full_results.append({
            "image_path": p.image_path[0],
            "category": CATEGORY,
            "gt_label": bool(p.gt_label.item()),
            "pred_score": float(p.pred_score.item()),
            "regions": regions,
        })
    return full_results


def save_results(full_results):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(full_results, f, indent=2)


def main():
    datamodule = build_datamodule()
    model = load_model()
    predictions = run_predictions(model, datamodule)
    full_results = build_full_results(predictions, THRESHOLD)
    save_results(full_results)
    print(f"Saved {len(full_results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()