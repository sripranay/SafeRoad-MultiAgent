from typing import List, Dict
import numpy as np
from ultralytics import YOLO

# Labels
HAZARD_LABELS = {"pothole", "crack", "speed_breaker", "road_damage"}
PEDESTRIAN_LABELS = {"person"}
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}

class VisionAgent:
    def __init__(self, road_model_path: str, vehicle_model_path: str, conf: float = 0.35, iou: float = 0.5):
        self.road_model = YOLO(road_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.conf = conf
        self.iou = iou

    def _yolo_to_detections(self, result, label_map=None) -> List[Dict]:
        dets = []
        if result is None or result.boxes is None:
            return dets
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c in zip(xyxy, cls):
            label = label_map.get(c, str(c)) if label_map else str(c)
            dets.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label
            })
        return dets

    def detect(self, frame: np.ndarray) -> List[Dict]:
        road_res = self.road_model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        road_names = getattr(self.road_model, "names", None)
        road_label_map = {int(k): v for k, v in road_names.items()} if isinstance(road_names, dict) else None
        road_dets = self._yolo_to_detections(road_res, road_label_map)

        # Normalize labels
        for d in road_dets:
            lbl = d["label"].lower()
            if "speed" in lbl and "break" in lbl:
                d["label"] = "speed_breaker"
            elif "poth" in lbl:
                d["label"] = "pothole"
            elif "crack" in lbl:
                d["label"] = "crack"
            elif "damage" in lbl:
                d["label"] = "road_damage"

        veh_res = self.vehicle_model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        veh_label_map = self.vehicle_model.names if isinstance(self.vehicle_model.names, dict) else None
        veh_dets = self._yolo_to_detections(veh_res, veh_label_map)

        filtered = []
        for d in veh_dets:
            lbl = d["label"].lower()
            if lbl in PEDESTRIAN_LABELS or lbl in VEHICLE_LABELS:
                filtered.append(d)

        return road_dets + filtered
