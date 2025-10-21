```python
# app.py - Final cleaned version with fixes for Ultralytics warning,
# deprecation update, defensive monkeypatches for streamlit-webrtc/aioice,
# and logging for easier debugging.

import os
import cv2
import tempfile
import numpy as np
import logging
import streamlit as st

# Ensure Ultralytics writes to a tmp folder on Streamlit Cloud (silences warning)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# Async debug (helps produce better stack traces)
os.environ.setdefault("PYTHONASYNCIODEBUG", "1")

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("streamlit_webrtc").setLevel(logging.DEBUG)
logging.getLogger("aioice").setLevel(logging.DEBUG)
logging.getLogger("aiortc").setLevel(logging.DEBUG)

# Defensive monkeypatch: make selector datagram transport no-op if socket closed.
# This avoids AttributeError: 'NoneType' object has no attribute 'sendto'
try:
    import asyncio
    import types
    from functools import wraps
    sel_mod = asyncio.selector_events

    if hasattr(sel_mod, "_SelectorDatagramTransport"):
        _TransportCls = sel_mod._SelectorDatagramTransport
        _orig_sendto = getattr(_TransportCls, "sendto", None)

        if _orig_sendto is not None:
            @wraps(_orig_sendto)
            def _safe_sendto(self, data, addr=None):
                try:
                    sock = getattr(self, "_sock", None)
                    loop = getattr(self, "_loop", None)
                    if sock is None or loop is None:
                        logging.debug("Datagram transport or loop is None; skipping sendto")
                        return
                    return _orig_sendto(self, data, addr)
                except Exception:
                    logging.exception("safe_sendto: suppressed exception while sending datagram")

            _TransportCls.sendto = _safe_sendto
except Exception:
    logging.exception("Failed to apply safe_sendto monkeypatch")

# Defensive monkeypatch: avoid ShutdownObserver.stop calling is_alive on None
try:
    from streamlit_webrtc import shutdown as _webrtc_shutdown

    def _safe_shutdown_stop(self):
        """Safely stop polling thread without crashing if thread is None."""
        try:
            t = getattr(self, "_polling_thread", None)
            if t is not None and hasattr(t, "is_alive"):
                try:
                    if t.is_alive():
                        # small non-blocking join attempt
                        t.join(timeout=0.01)
                except Exception:
                    logging.debug("Join on polling thread failed", exc_info=True)
        except Exception:
            logging.exception("safe shutdown stop failed")

    _webrtc_shutdown.ShutdownObserver.stop = _safe_shutdown_stop
except Exception:
    logging.exception("Failed to monkeypatch streamlit_webrtc.shutdown")

# ---------- YOLO imports & models ----------
from ultralytics import YOLO

# Load models (paths should exist in your repo)
road_model = YOLO("models/best.pt")        # custom trained model (road damages, speed breakers)
vehicle_model = YOLO("models/yolov8s.pt")  # pretrained YOLOv8s for vehicles + people

# ---------- Detection helper ----------
def run_detection(image, use_road_model=True, use_vehicle_model=True):
    annotated = image.copy()

    if use_road_model:
        try:
            results = road_model(image)
            for r in results[0].boxes:
                cls_id = int(r.cls)
                label = road_model.names.get(cls_id, str(cls_id))
                xyxy = r.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception:
            logging.exception("Error running road_model")

    if use_vehicle_model:
        try:
            results = vehicle_model(image)
            for r in results[0].boxes:
                cls_id = int(r.cls)
                label = vehicle_model.names.get(cls_id, str(cls_id))
                xyxy = r.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception:
            logging.exception("Error running vehicle_model")

    return annotated

# ---------- WebRTC transformer ----------
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration,
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = run_detection(img, use_road_model=True, use_vehicle_model=True)
            return img
        except Exception:
            logging.exception("Error inside VideoTransformer.transform")
            try:
                return frame.to_ndarray(format="bgr24")
            except Exception:
                return np.zeros((480, 640, 3), dtype=np.uint8)

# ---------- Streamlit UI ----------
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
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    try:
        webrtc_streamer(
            key="road-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            # Updated API: use video_processor_factory (replaces deprecated video_transformer_factory)
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception:
        logging.exception("Error starting webrtc_streamer")
```
