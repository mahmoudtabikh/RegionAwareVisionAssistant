import torch
import json
import os
import numpy as np
import onnxruntime as ort
from src.lib.inference import load_onnx_session, run_onnx_inference
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode

# ===== CONFIGURATION =====
CATEGORY = "leather"  # change to "leather" for the other model
MVTECAD_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
CKPT_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"
ONNX_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{CATEGORY}/weights/onnx/model.onnx"
EXPORT_DIR = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{CATEGORY}/weights/onnx"

NUM_SAMPLES = 10  # how many test images to compare
VAL_SPLIT_RATIO = 0.2
SEED = 42
# ==========================

torch.set_float32_matmul_precision("medium")


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


def compare_predictions(ckpt_predictions, onnx_session):
    comparison = []

    for p in ckpt_predictions:
        ckpt_score = float(p.pred_score.item())
        image_tensor = p.image

        onnx_outputs = run_onnx_inference(
            onnx_session,
            image_tensor,
        )

        onnx_score = float(onnx_outputs["pred_score"].squeeze())

        # ===== ANOMALY MAP =====
        ckpt_map = p.anomaly_map.cpu().numpy()
        onnx_map = onnx_outputs["anomaly_map"]

        map_abs_diff = np.abs(ckpt_map - onnx_map)

        map_max_abs_diff = float(map_abs_diff.max())
        map_mean_abs_diff = float(map_abs_diff.mean())

        # ===== PREDICTION MASK =====
        ckpt_mask = p.pred_mask.cpu().numpy()
        onnx_mask = onnx_outputs["pred_mask"]

        mask_disagreeing_pixels = int(
            (ckpt_mask != onnx_mask).sum()
        )

        mask_disagreement_percentage = (
            mask_disagreeing_pixels / ckpt_mask.size
        ) * 100.0

        comparison.append(
            {
                "image_path": p.image_path[0],

                "ckpt_score": ckpt_score,
                "onnx_score": onnx_score,
                "score_abs_diff": abs(ckpt_score - onnx_score),

                "map_max_abs_diff": map_max_abs_diff,
                "map_mean_abs_diff": map_mean_abs_diff,

                "mask_disagreeing_pixels": mask_disagreeing_pixels,
                "mask_disagreement_percentage": mask_disagreement_percentage,
            }
        )

    return comparison


def build_summary(comparison):
    summary = {
        "category": CATEGORY,
        "num_samples": len(comparison),

        # Score differences
        "score_max_abs_diff": max(
            item["score_abs_diff"]
            for item in comparison
        ),
        "score_mean_abs_diff": (
            sum(
                item["score_abs_diff"]
                for item in comparison
            )
            / len(comparison)
        ),

        # Anomaly map differences
        "map_max_abs_diff": max(
            item["map_max_abs_diff"]
            for item in comparison
        ),
        "map_mean_abs_diff": (
            sum(
                item["map_mean_abs_diff"]
                for item in comparison
            )
            / len(comparison)
        ),

        # Prediction mask differences
        "mask_max_disagreement_percentage": max(
            item["mask_disagreement_percentage"]
            for item in comparison
        ),
        "mask_mean_disagreement_percentage": (
            sum(
                item["mask_disagreement_percentage"]
                for item in comparison
            )
            / len(comparison)
        ),
    }

    return summary


def save_results(comparison):
    os.makedirs(EXPORT_DIR, exist_ok=True)

    summary = build_summary(comparison)

    results = {
        "summary": summary,
        "per_sample": comparison,
    }

    output_path = f"{EXPORT_DIR}/sanity_check_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    return output_path


def main():
    print(f"Comparing checkpoint vs ONNX for category: {CATEGORY}")

    datamodule = build_datamodule()
    ckpt_model = load_ckpt_model()

    ckpt_predictions = get_ckpt_predictions(
        ckpt_model,
        datamodule,
        NUM_SAMPLES,
    )

    onnx_session = load_onnx_session(CATEGORY)

    print(
        "ONNX input info:",
        [(i.name, i.shape) for i in onnx_session.get_inputs()],
    )

    print(
        "ONNX output info:",
        [(o.name, o.shape) for o in onnx_session.get_outputs()],
    )

    comparison = compare_predictions(
        ckpt_predictions,
        onnx_session,
    )

    # ===== PER-SAMPLE RESULTS =====
    for c in comparison:
        print(c)

    # ===== SUMMARY =====
    summary = build_summary(comparison)

    print("\n===== SANITY CHECK SUMMARY =====")
    print(f"Category: {summary['category']}")
    print(f"Samples: {summary['num_samples']}")

    print("\nScore:")
    print(f"  Max absolute difference:  {summary['score_max_abs_diff']:.8e}")
    print(f"  Mean absolute difference: {summary['score_mean_abs_diff']:.8e}")

    print("\nAnomaly map:")
    print(f"  Max absolute difference:  {summary['map_max_abs_diff']:.8e}")
    print(f"  Mean absolute difference: {summary['map_mean_abs_diff']:.8e}")

    print("\nPrediction mask:")
    print(
        f"  Max disagreement:  "
        f"{summary['mask_max_disagreement_percentage']:.6f}%"
    )
    print(
        f"  Mean disagreement: "
        f"{summary['mask_mean_disagreement_percentage']:.6f}%"
    )

    # ===== SAVE RELEASE SANITY CHECK =====
    output_path = save_results(comparison)

    print(f"\nSanity check results saved to: {output_path}")


if __name__ == "__main__":
    main()
