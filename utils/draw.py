import cv2
import numpy as np
from typing import List, Dict

BOX_COLOR = (30, 144, 255)
TXT_COLOR = (255, 255, 255)

def draw_boxes(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d["label"].upper()
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TXT_COLOR, 2)
    return img
