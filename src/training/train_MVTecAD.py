# 1. Import required modules
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torch

torch.set_float32_matmul_precision('medium')

MVTecAD_dir="/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD"
imagenet_dir = "/home/mahmoud/projects/RegionAwareVisionAssistant/data/imagenette"
# 2. Create a dataset
# MVTecAD is a popular dataset for anomaly detection
datamodule = MVTecAD(
    root=MVTecAD_dir,  # Path to download/store the dataset
    category="wood",  # MVTec category to use
    train_batch_size=1,  # Number of images per training batch
    eval_batch_size=1,  # Number of images per validation/test batch
)

# 3. Initialize the model
model = EfficientAd(
    imagenet_dir = imagenet_dir,
    model_size="small",
)

# 4. Create the training engine
engine = Engine(max_steps=70000)
# 5. Train the model
# This produces a lightning model (.ckpt)
engine.fit(datamodule=datamodule, model=model)

# 6. Test the model performance
test_results = engine.test(datamodule=datamodule, model=model)
