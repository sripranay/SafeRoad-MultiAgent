import os
from enum import Enum
from typing import List, Dict

class LLMMode(str, Enum):
    OFFLINE = "offline"
    OPENAI = "openai"
    GEMINI = "gemini"

class LLMAgent:
    def __init__(self, mode: LLMMode = LLMMode.OFFLINE):
        self.mode = mode
        self._init_clients()

    def _init_clients(self):
        self.oai_client = None
        self.gem_client = None
        if self.mode == LLMMode.OPENAI and os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self.oai_client = OpenAI()
        elif self.mode == LLMMode.GEMINI and os.getenv("GEMINI_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gem_client = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.mode = LLMMode.OFFLINE

    def _offline_alert(self, risk: Dict) -> str:
        parts = []
        counts = risk.get("counts", {})
        if counts.get("hazards_center", 0): parts.append("Hazard ahead.")
        if counts.get("pedestrians", 0): parts.append("Pedestrians nearby.")
        if counts.get("vehicles", 0) >= 2: parts.append("Heavy traffic.")
        if not parts: parts.append("Road clear.")
        return f"{risk['level']} RISK: " + " ".join(parts)

    def generate_alert(self, detections: List[Dict], risk: Dict) -> str:
        if self.mode == LLMMode.OPENAI and self.oai_client:
            try:
                resp = self.oai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user",
                               "content": f"Generate a short road-safety warning for {risk}."}],
                    max_tokens=60,
                )
                return resp.choices[0].message.content.strip()
            except: return self._offline_alert(risk)

        if self.mode == LLMMode.GEMINI and self.gem_client:
            try:
                resp = self.gem_client.generate_content(
                    f"Generate a short road-safety warning for {risk}."
                )
                return resp.text.strip()
            except: return self._offline_alert(risk)

        return self._offline_alert(risk)
