import torch
import cv2
import json
import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode

torch.set_float32_matmul_precision('medium')

MVTecAD_dir="/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
imagenet_dir = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
model_path = "/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/v0/weights/lightning/model.ckpt"
# 2. Create a dataset
# MVTecAD is a popular dataset for anomaly detection
datamodule = MVTecAD(
    root=MVTecAD_dir,  # Path to download/store the dataset
    category="leather",  # MVTec category to use
    train_batch_size=1,  # Number of images per training batch
    eval_batch_size=1,  # Number of images per validation/test batch
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.2,  # Ratio of test images to use for validation
    seed=42,  # Random seed for reproducibility

)

model = EfficientAd.load_from_checkpoint(
    checkpoint_path=model_path,
    imagenet_dir=imagenet_dir,
    model_size="small",
)
engine = Engine()
datamodule.setup(stage="test")


predictions = engine.predict(
    model=model,
    dataloaders=datamodule.test_dataloader(),
    ckpt_path=model_path,
)
threshold = 0.5046368241  # leather's threshold, from your saved metrics

full_results = []
for p in predictions:  # predictions from leather's val set — confirm you're using leather's predict() output here, not leather's leftover variable
    anomaly_map_np = p.anomaly_map.cpu().numpy().squeeze()
    binary_mask = (anomaly_map_np >= threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        compactness = area / bbox_area if bbox_area > 0 else 0
        regions.append({
            "region_id": f"region_{i+1}",
            "polygon": contour.squeeze().tolist(),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(area),
            "compactness": float(compactness),
        })

    full_results.append({
        "image_path": p.image_path[0],
        "category": "leather",
        "gt_label": bool(p.gt_label.item()),
        "pred_score": float(p.pred_score.item()),
        "regions": regions,
    })

with open("/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_test_full.json", "w") as f:
    json.dump(full_results, f, indent=2)
