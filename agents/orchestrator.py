from typing import Dict

class Orchestrator:
    def __init__(self, vision, risk, llm):
        self.vision = vision
        self.risk = risk
        self.llm = llm

    def run_once(self, frame) -> Dict:
        detections = self.vision.detect(frame)
        risk_out = self.risk.assess(detections, frame.shape)
        alert = self.llm.generate_alert(detections, risk_out)
        return {"detections": detections, "risk": risk_out, "alert": alert}
