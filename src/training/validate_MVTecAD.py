import torch
import json
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode

# ===== CONFIGURATION =====
CATEGORY = "wood"  # change to "leather" for the other model
MVTECAD_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
MODEL_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"
OUTPUT_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/{CATEGORY}_val_scores.json"
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
    datamodule.setup(stage="predict")

    predictions = engine.predict(
        model=model,
        dataloaders=datamodule.val_dataloader(),
        ckpt_path=MODEL_PATH,
    )
    return predictions


def build_results_summary(predictions):
    results_summary = []
    if predictions is not None:
        for p in predictions:
            results_summary.append({
                "image_path": p.image_path[0],
                "gt_label": bool(p.gt_label.item()),
                "pred_score": float(p.pred_score.item()),
                "pred_label": bool(p.pred_label.item()),
            })
    return results_summary


def save_results(results_summary):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results_summary, f, indent=2)


def main():
    datamodule = build_datamodule()
    model = load_model()
    predictions = run_predictions(model, datamodule)
    results_summary = build_results_summary(predictions)
    save_results(results_summary)
    print(f"Saved {len(results_summary)} val results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()