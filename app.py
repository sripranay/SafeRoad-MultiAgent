import os
import logging
import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
# put this at very top of app.py (before streamlit_webrtc is used)
import logging

# Monkey-patch ShutdownObserver.stop to be defensive if _polling_thread is None
try:
    from streamlit_webrtc import shutdown as _webrtc_shutdown

    def _safe_shutdown_stop(self):
        try:
            t = getattr(self, "_polling_thread", None)
            # If thread exists and has is_alive, check it; otherwise skip safely
            if t is not None and hasattr(t, "is_alive"):
                if t.is_alive():
                    # try join briefly (non-blocking safeguard)
                    try:
                        t.join(timeout=0.01)
                    except Exception:
                        logging.debug("join on polling thread failed", exc_info=True)
        except Exception:
            logging.exception("safe shutdown stop failed")

    # Replace the method
    _webrtc_shutdown.ShutdownObserver.stop = _safe_shutdown_stop
except Exception:
    logging.exception("Failed to monkeypatch streamlit_webrtc.shutdown")

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration,
)

# -------------------------------
# Logging & Async debug
# -------------------------------
# (helps show more useful traces in Cloud logs)
os.environ.setdefault("PYTHONASYNCIODEBUG", "1")
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("streamlit_webrtc").setLevel(logging.DEBUG)
logging.getLogger("aioice").setLevel(logging.DEBUG)
logging.getLogger("aiortc").setLevel(logging.DEBUG)

# -------------------------------
# Load models
# -------------------------------
road_model = YOLO("models/best.pt")
vehicle_model = YOLO("models/yolov8s.pt")

# -------------------------------
# Helper: Run Detection
# -------------------------------
def run_detection(image, use_road_model=True, use_vehicle_model=True):
    annotated = image.copy()

    if use_road_model:
        results = road_model(image)
        for r in results[0].boxes:
            cls_id = int(r.cls)
            label = road_model.names[cls_id]
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
            cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if use_vehicle_model:
        results = vehicle_model(image)
        for r in results[0].boxes:
            cls_id = int(r.cls)
            label = vehicle_model.names[cls_id]
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return annotated

# -------------------------------
# WebRTC Live Camera transformer
# -------------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = run_detection(img, use_road_model=True, use_vehicle_model=True)
            return img
        except Exception:
            logging.exception("Error inside VideoTransformer.transform")
            # return the original frame (or a black frame) to avoid crashing transport
            try:
                return frame.to_ndarray(format="bgr24")
            except Exception:
                # final fallback: return a small black image
                return np.zeros((480, 640, 3), dtype=np.uint8)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Road Detection App", layout="wide")
st.title("🚦 Road Detection App - Vehicles, Humans & Damages")

option = st.sidebar.radio("Choose Input", ["Upload Image", "Upload Video", "Live Camera"])

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        annotated = run_detection(image)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = run_detection(frame)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

elif option == "Live Camera":
    # use a stable RTCConfiguration with STUN; add TURN if you have it
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # keep webrtc_ctx reference in case we need to stop later
    webrtc_ctx = None
    try:
        webrtc_ctx = webrtc_streamer(
            key="road-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,  # run transformer in background thread/coroutine
        )
    except Exception:
        logging.exception("Error starting webrtc_streamer")
    # no explicit stop here; Streamlit will handle session lifecycles,
    # but app-level guards are below if you extend this file.
