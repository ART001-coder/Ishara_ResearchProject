import os, sys, queue, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from src.utils.config import load_config
from src.inference.predictor import RealtimePredictor
from src.inference.gloss_buffer import GlossBuffer
from src.inference.sentence_builder import SentenceBuilder

st.set_page_config(page_title="Ishara — ISL Translator", layout="wide")

cfg = load_config("config.yaml")
result_queue: "queue.Queue[tuple]" = queue.Queue()


@st.cache_resource
def get_predictor():
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
        img = frame.to_ndarray(format="bgr24")
        annotated, word, conf = self.predictor.process_frame(img)

        with self.lock:
            if self.buffer.add_prediction(word, conf):
                result_queue.put(("glosses", list(self.buffer.glosses)))
            if self.buffer.should_send():
                sentence = self.builder.build_sentence(self.buffer.flush())
                result_queue.put(("sentence", sentence))

        return av.VideoFrame.from_ndarray(annotated, format="rgb24")


st.title("Ishara — ISL to sentence translator")

ctx = webrtc_streamer(
    key="ishara",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

gloss_box = st.empty()
sentence_box = st.empty()

if ctx.state.playing:
    while True:
        kind, payload = result_queue.get()
        if kind == "glosses":
            with gloss_box.container():
                for w, c in payload:
                    st.progress(min(1.0, max(0.0, c)), text=f"**{w.upper()}** ({c*100:.1f}%)")
        else:
            sentence_box.markdown(f"### \"{payload}\"")