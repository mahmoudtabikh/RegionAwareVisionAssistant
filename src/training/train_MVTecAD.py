from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torch

# ===== CONFIGURATION =====
CATEGORY = "wood"  # change to "leather" for the other model
MVTECAD_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
MAX_STEPS = 70000
MODEL_SIZE = "small"
# ==========================

torch.set_float32_matmul_precision('medium')


def build_datamodule():
    datamodule = MVTecAD(
        root=MVTECAD_DIR,
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=1,
    )
    return datamodule


def build_model():
    model = EfficientAd(
        imagenet_dir=IMAGENET_DIR,
        model_size=MODEL_SIZE,
    )
    return model


def train_and_test(model, datamodule):
    engine = Engine(max_steps=MAX_STEPS)
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)
    return test_results


def main():
    datamodule = build_datamodule()
    model = build_model()
    test_results = train_and_test(model, datamodule)
    print(test_results)


if __name__ == "__main__":
    main()