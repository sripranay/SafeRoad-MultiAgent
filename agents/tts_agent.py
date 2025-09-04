import tempfile
from gtts import gTTS
import streamlit as st

class TTSAgent:
    def __init__(self, lang="en"):
        self.lang = lang

    def speak(self, text: str):
        try:
            tts = gTTS(text=text, lang=self.lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.warning(f"TTS failed: {e}")
