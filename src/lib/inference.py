import onnxruntime as ort
import cv2
import torch
from torchvision.transforms.v2 import Resize


def process_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array (HWC) to torch tensor (BCHW) with batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Scale to [0, 1]
    image_tensor = image_tensor / 255.0
    
    # Apply resize with antialias (matches anomalib's preprocessing)
    resize_transform = Resize((256, 256), antialias=True)
    image_tensor = resize_transform(image_tensor)
    
    # Convert back to numpy for ONNX Runtime (BCHW format, [0, 1] range)
    image = image_tensor.numpy()
    
    return image

def load_onnx_session(category):
    if category not in ["leather", "wood"]:
        raise ValueError("Invalid category. Must be 'leather' or 'wood'.")
    onnx_path = f"/home/mahmoud/projects/RegionAwareVisionAssistant/results/exports/{category}/weights/onnx/model.onnx"
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    return session


def run_onnx_inference(session, image):
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    ort_inputs = {
        input_name: image
    }

    ort_outputs = session.run(output_names, ort_inputs)

    return dict(zip(output_names, ort_outputs))
