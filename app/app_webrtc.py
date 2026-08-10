"""
==============================================================================
Module: app/app_webrtc.py
Role: Browser-sourced WebRTC Streamlit UI for hosted deployment.

Why this exists:
    app/streamlit_app.py uses cv2.VideoCapture, which opens a camera on the
    machine running Python. That is the same machine as the browser when run
    locally, but on a hosted server there is no camera attached at all.
    This module streams frames from the user's browser instead.

    src/ is unchanged - only the frame source differs.
==============================================================================
"""

import os
import sys
import queue
import threading

# Make 'src.*' importable regardless of the working directory streamlit
# was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from src.utils.config import load_config
from src.inference.predictor import RealtimePredictor
from src.inference.gloss_buffer import GlossBuffer
from src.inference.sentence_builder import SentenceBuilder


st.set_page_config(page_title="Ishara - ISL Translator", layout="wide")

cfg = load_config("config.yaml")

# Frames are processed on a worker thread that cannot touch st.session_state,
# so results are handed back to the UI thread through this queue.
result_queue: "queue.Queue[tuple]" = queue.Queue()


@st.cache_resource
def get_predictor():
    """Load MediaPipe + the Bi-LSTM once per container, not per reconnect."""
    return RealtimePredictor(config_path="config.yaml")


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.predictor = get_predictor()
        self.buffer = GlossBuffer(
            conf_threshold=cfg['inference'].get('confidence_threshold', 0.15),
            min_glosses=cfg['inference'].get('buffer_min_glosses', 3),
            timeout_sec=cfg['inference'].get('buffer_timeout_sec', 5.0),
        )
        self.builder = SentenceBuilder(
            timeout=cfg['llm'].get('api_timeout_sec', 3.0),
            use_fallback=cfg['llm'].get('enable_template_fallback', True),
        )
        self.lock = threading.Lock()

    def recv(self, frame):
        # process_frame expects BGR and returns an annotated RGB frame.
        img = frame.to_ndarray(format="bgr24")
        annotated, word, conf = self.predictor.process_frame(img)

        with self.lock:
            if self.buffer.add_prediction(word, conf):
                result_queue.put(("glosses", list(self.buffer.glosses)))

            if self.buffer.should_send():
                sentence = self.builder.build_sentence(self.buffer.flush())
                result_queue.put(("sentence", sentence))

        return av.VideoFrame.from_ndarray(annotated, format="rgb24")


st.title("Ishara - ISL to sentence translator")
st.caption("Allow camera access, then sign. Glosses accumulate below and are "
           "reconstructed into a sentence once enough are detected.")

ctx = webrtc_streamer(
    key="ishara",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    # Drop frames when inference lags instead of queueing them, so the video
    # does not drift seconds behind the signer.
    async_processing=True,
)

st.subheader("Detected glosses")
gloss_box = st.empty()

st.subheader("Reconstructed sentence")
sentence_box = st.empty()

if ctx.state.playing:
    while True:
        kind, payload = result_queue.get()

        if kind == "glosses":
            with gloss_box.container():
                for word, conf in payload:
                    st.progress(
                        min(1.0, max(0.0, conf)),
                        text=f"**{word.upper()}** ({conf * 100:.1f}%)",
                    )
        else:
            sentence_box.markdown(f'### "{payload}"')