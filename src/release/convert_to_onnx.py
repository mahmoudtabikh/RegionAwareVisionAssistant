from anomalib.deploy import ExportType
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# ===== CONFIGURATION =====
CATEGORY = "wood"  # change to "wood" for the other model
MODEL_PATH = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/EfficientAd/MVTecAD/{CATEGORY}/v0/weights/lightning/model.ckpt"
IMAGENET_DIR = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
INPUT_SIZE = [256, 256]
EXPORT_DIR = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{CATEGORY}"

# ==== MAIN FUNCTION =====
def main():
    model = EfficientAd.load_from_checkpoint(
        checkpoint_path=MODEL_PATH,
        imagenet_dir=IMAGENET_DIR,
        model_size="small",
    )
    engine = Engine()

    onnx_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        input_size=INPUT_SIZE,
        export_root=EXPORT_DIR
    )
    print(onnx_path)

# === ENTRY POINT =====
if __name__ == "__main__":
    main()
