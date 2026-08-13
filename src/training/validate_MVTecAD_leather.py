import torch
import json
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import TorchInferencer

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
datamodule.setup(stage="predict")

predictions = engine.predict(
    model=model,
    dataloaders=datamodule.val_dataloader(),
    ckpt_path=model_path,
)
results_summary = []
if predictions is not None:
    for p in predictions:
        results_summary.append({
            "image_path": p.image_path[0],
            "gt_label": bool(p.gt_label.item()),
            "pred_score": float(p.pred_score.item()),
            "pred_label": float(p.pred_label.item()),
            "anomaly_map": float(p.anomaly_map.item()),
    })

with open("/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/leather/leather_val_scores.json", "w") as f:
    json.dump(results_summary, f, indent=2)
