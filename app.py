import os
import cv2
import av
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Local imports
from agents.vision_agent import VisionAgent
from agents.risk_agent import RiskAgent
from agents.llm_agent import LLMAgent, LLMMode
from agents.orchestrator import Orchestrator
from agents.tts_agent import TTSAgent
from utils.draw import draw_boxes

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="SafeRoad Multi-Agent Assistant", page_icon="🛣️", layout="wide")
st.title("🛣️ SafeRoad Multi-Agent Assistant (SMAA)")
st.caption("Vision + Risk + LLM + TTS Agents working together for road safety.")

# -------------------------------
# Sidebar Settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    llm_mode = st.selectbox("LLM Mode", ["OFFLINE", "OPENAI", "GEMINI"], index=0)
    conf_thres = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)
    iou_thres = st.slider("NMS IoU", 0.2, 0.9, 0.5, 0.05)
    st.write("Make sure models are in ./models (best.pt, yolov8s.pt)")

# -------------------------------
# Initialize Agents
# -------------------------------
vision = VisionAgent("models/best.pt", "models/yolov8s.pt", conf=conf_thres, iou=iou_thres)
risk = RiskAgent()
mode_map = {"OFFLINE": LLMMode.OFFLINE, "OPENAI": LLMMode.OPENAI, "GEMINI": LLMMode.GEMINI}
llm = LLMAgent(mode=mode_map[llm_mode])
tts = TTSAgent()
orchestrator = Orchestrator(vision, risk, llm)

# -------------------------------
# Helper Functions
# -------------------------------
def process_frame(frame):
    detections = vision.detect(frame)
    risk_out = risk.assess(detections, frame.shape)
    alert_text = llm.generate_alert(detections, risk_out)
    annotated = draw_boxes(frame.copy(), detections)
    return annotated, risk_out, alert_text

# -------------------------------
# Tabs: Image | Video | Webcam
# -------------------------------
img_tab, vid_tab, cam_tab = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Webcam"])

# ---------- Image Tab ----------
with img_tab:
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file).convert("RGB"))
        annotated, risk_out, alert_text = process_frame(image)
        st.image(annotated, caption="Annotated", use_column_width=True)
        st.metric("Risk Level", risk_out["level"], help=risk_out.get("reason"))
        st.text_area("Alert", value=alert_text, height=120)
        tts.speak(alert_text)

# ---------- Video Tab ----------
with vid_tab:
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        last_alert, last_risk = "", {"level": "LOW"}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, risk_out, alert_text = process_frame(frame)
            last_alert, last_risk = alert_text, risk_out
        cap.release()

        st.video(tfile.name)
        st.metric("Risk Level", last_risk["level"], help=last_risk.get("reason"))
        st.text_area("Alert", value=last_alert, height=120)
        tts.speak(last_alert)

# ---------- Webcam Tab ----------
with cam_tab:
    st.write("Live detection from your webcam with voice alerts.")

    RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, risk_out, alert_text = process_frame(img)
        tts.speak(alert_text)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="road-safety",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
    )
