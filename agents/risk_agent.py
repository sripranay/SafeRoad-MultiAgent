from typing import List, Dict

CENTER_ROI = 0.35  # fraction of width/height for center box

class RiskAgent:
    def assess(self, detections: List[Dict], frame_shape) -> Dict:
        H, W = frame_shape[0], frame_shape[1]
        cx1 = int(W*(0.5 - CENTER_ROI/2)); cy1 = int(H*(0.6 - CENTER_ROI/2))
        cx2 = int(W*(0.5 + CENTER_ROI/2)); cy2 = int(H*(0.6 + CENTER_ROI/2))

        hazards, hazards_center, pedestrians, vehicles = 0, 0, 0, 0

        def in_center(b):
            x1, y1, x2, y2 = b
            inter_x1, inter_y1 = max(x1, cx1), max(y1, cy1)
            inter_x2, inter_y2 = min(x2, cx2), min(y2, cy2)
            return (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

        for d in detections:
            lbl = d["label"].lower()
            if lbl in {"pothole", "crack", "speed_breaker", "road_damage"}:
                hazards += 1
                if in_center(d["bbox"]): hazards_center += 1
            elif lbl == "person":
                pedestrians += 1
            elif lbl in {"car", "bus", "truck", "motorcycle", "bicycle"}:
                vehicles += 1

        if hazards_center >= 1 and (pedestrians + vehicles) >= 2:
            level, reason = "HIGH", "Hazard in path with traffic"
        elif hazards_center >= 1:
            level, reason = "MEDIUM", "Hazard in driving path"
        elif hazards >= 1 and (pedestrians + vehicles) >= 2:
            level, reason = "MEDIUM", "Hazard nearby with traffic"
        else:
            level, reason = "LOW", "No immediate hazard"

        return {"level": level, "reason": reason, "counts": {
            "hazards": hazards,
            "hazards_center": hazards_center,
            "pedestrians": pedestrians,
            "vehicles": vehicles
        }}
