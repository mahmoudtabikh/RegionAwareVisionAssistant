import cv2
import numpy as np

def extract_regions(anomaly_map_np, threshold):
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
    return regions
