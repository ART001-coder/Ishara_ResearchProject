"""
==============================================================================
Module: app/streamlit_app.py
Role: Main Streamlit Application UI for Real-Time ISL Translation (Premium Design)
Reference: implementation_plan.md -> Section 1.6, 4.10, Section 8.3 (Bug I4), Day 2/8/11
==============================================================================
"""

import os
import sys
import time

# Ensure the project root (parent of this app/ folder) is importable as 'src.*',
# regardless of the working directory `streamlit run` was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import streamlit as st
from src.utils.config import load_config
from src.inference.predictor import RealtimePredictor
from src.inference.gloss_buffer import GlossBuffer
from src.inference.sentence_builder import SentenceBuilder


def initialize_session_state():
    """
    Initializes st.session_state persistence to prevent video stream restart on UI clicks (Bug I4).
    """
    if 'initialized' not in st.session_state:
        st.session_state.cfg = load_config("config.yaml")
        st.session_state.predictor = None  # Lazy-loaded on "Start Translation"
        st.session_state.buffer = GlossBuffer(
            conf_threshold=st.session_state.cfg['inference'].get('confidence_threshold', 0.15),
            min_glosses=st.session_state.cfg['inference'].get('buffer_min_glosses', 3),
            timeout_sec=st.session_state.cfg['inference'].get('buffer_timeout_sec', 5.0)
        )
        st.session_state.sentence_builder = SentenceBuilder(
            timeout=st.session_state.cfg['llm'].get('api_timeout_sec', 3.0),
            use_fallback=st.session_state.cfg['llm'].get('enable_template_fallback', True)
        )

        # Preload demo sentences into cache for zero-latency presentation response (Section 11.2 R6)
        st.session_state.sentence_builder.preload_cache([
            ([("doctor", 0.92), ("need", 0.85), ("help", 0.78)], "I need help from the doctor."),
            ([("stomach", 0.91), ("pain", 0.95), ("bad", 0.80)], "I have bad stomach pain."),
            ([("hello", 0.99), ("name", 0.88), ("what", 0.75)], "Hello, what is your name?"),
            ([("medicine", 0.88), ("when", 0.72), ("see", 0.65)], "When should I take the medicine?"),
            ([("water", 0.90), ("drink", 0.82), ("please", 0.95)], "Can I please have some water to drink?")
        ])

        st.session_state.is_running = False
        st.session_state.gloss_history = []
        st.session_state.translated_sentence = ""
        st.session_state.initialized = True


def get_working_webcam(camera_index: int = 0):
    """
    Attempts to open webcam using multiple backends (DirectShow for Windows, default, MSMF).
    """
    # 1. Try DirectShow backend on Windows (fastest opening)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()

    # 2. Try default backend
    cap = cv2.VideoCapture(camera_index)
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()

    # 3. Try alternative camera index 1
    if camera_index == 0:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()

    return None


def inject_custom_css():
    """
    Injects modern CSS styling, glassmorphism cards, gradient highlights, and responsive layouts.
    """
    st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #0d1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 6px;
    }
    
    /* Panel Cards */
    .panel-card {
        background: rgba(31, 41, 55, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .panel-header {
        color: #f3f4f6;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Status Badges */
    .badge-live {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-standby {
        background: rgba(107, 114, 128, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(107, 114, 128, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Output Sentence Box */
    .sentence-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 18px 20px;
        color: #ecfdf5;
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    .sentence-placeholder {
        background: rgba(55, 65, 81, 0.3);
        border: 1px dashed rgba(156, 163, 175, 0.3);
        border-radius: 12px;
        padding: 18px 20px;
        color: #9ca3af;
        font-size: 1rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Ishara - ISL Translator",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()
    initialize_session_state()

    # App Header Card
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🤟 Ishara: ISL to Sentence Translator</div>
        <div class="header-subtitle">Real-Time Indian Sign Language Recognition & Gemini LLM Reconstruction | Hospital & Reception Desk Assistance</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls & Information
    with st.sidebar:
        st.header("⚙️ Control Panel")
        cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        conf_thresh = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.15, step=0.05)
        st.session_state.cfg['inference']['confidence_threshold'] = conf_thresh
        if st.session_state.predictor is not None:
            st.session_state.predictor.cfg['inference']['confidence_threshold'] = conf_thresh
        st.session_state.buffer.conf_threshold = conf_thresh

        st.markdown("---")
        st.header("📌 System Information")
        st.markdown("""
        - **Pipeline:** MediaPipe → Bi-LSTM → Gemini 2.0 Flash
        - **Vocabulary:** 65 ISL Words (Hospital Scenario)
        - **Feature Dims:** 225 Landmarks (Pose + Hands)
        - **Target Latency:** < 500ms / sign
        """)
        st.markdown("---")
        st.caption("Ishara Project v1.0 | 2-Week Sprint Prototype")

    # Main Layout Columns
    col_video, col_output = st.columns([3, 2], gap="large")

    with col_video:
        # Video Container Header with Live Status Badge
        status_html = '<span class="badge-live">🟢 LIVE TRANSLATING</span>' if st.session_state.is_running else '<span class="badge-standby">⏸ STANDBY</span>'
        st.markdown(f"""
        <div class="panel-header">
            <span>📹 Live Video Stream & Skeleton Overlay</span>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

        # Control Buttons Placed ABOVE Video Box for Easy Access
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶ Start Translation", use_container_width=True, type="primary"):
                st.session_state.is_running = True
        with c2:
            if st.button("⏹ Stop Translation", use_container_width=True):
                st.session_state.is_running = False
        with c3:
            if st.button("🗑 Clear Buffer", use_container_width=True):
                st.session_state.buffer.clear()
                st.session_state.gloss_history = []
                st.session_state.translated_sentence = ""

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        frame_placeholder = st.empty()

        if not st.session_state.is_running:
            frame_placeholder.markdown("""
            <div style="background: rgba(17, 24, 39, 0.8); border: 2px dashed rgba(75, 85, 99, 0.4); border-radius: 12px; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #9ca3af;">
                <div style="font-size: 3rem; margin-bottom: 8px;">📹</div>
                <div style="font-size: 1.1rem; font-weight: 600;">Camera Stream Paused</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Click "▶ Start Translation" above to start recognition.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_output:
        st.markdown("""
        <div class="panel-header">
            <span>🤟 Detected Sign Glosses</span>
        </div>
        """, unsafe_allow_html=True)

        gloss_container = st.empty()

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel-header">
            <span>📝 Reconstructed English Sentence</span>
        </div>
        """, unsafe_allow_html=True)

        sentence_container = st.empty()

        if st.session_state.translated_sentence:
            sentence_container.markdown(f'<div class="sentence-box">"{st.session_state.translated_sentence}"</div>', unsafe_allow_html=True)
        else:
            sentence_container.markdown('<div class="sentence-placeholder">Awaiting sign gesture sequence...</div>', unsafe_allow_html=True)

    # Render current gloss history in right panel
    with gloss_container.container():
        if st.session_state.gloss_history:
            for word, conf in st.session_state.gloss_history:
                st.progress(min(1.0, max(0.0, conf)), text=f"**{word.upper()}** (Confidence: {conf*100:.1f}%)")
        else:
            st.markdown("<div style='color: #6b7280; font-style: italic; font-size: 0.9rem;'>No active glosses in buffer.</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Real-Time Video Stream Loop (when is_running is True)
    # --------------------------------------------------------------------------
    if st.session_state.is_running:
        if st.session_state.predictor is None:
            with st.spinner("Initializing AI Models (MediaPipe & PyTorch)..."):
                st.session_state.predictor = RealtimePredictor(config_path="config.yaml")

        cap = get_working_webcam(camera_index=cam_idx)

        if cap is None or not cap.isOpened():
            st.error(f"Error: Webcam device (Index {cam_idx}) unreachable or in use by another application. "
                     f"Please verify camera permissions or select another device index in the sidebar.")
            st.session_state.is_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Webcam stream disconnected or failed to read frame.")
                break

            # Run prediction pipeline frame by frame
            annotated_frame, word, confidence = st.session_state.predictor.process_frame(frame)
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            # Add valid prediction to deduplication gloss buffer
            ignored_labels = {"BUFFERING", "UNKNOWN", "IDLE", "IDLE (RAISE HANDS TO SIGN)", "NO HANDS DETECTED"}
            if word.upper() not in ignored_labels:
                st.session_state.buffer.add_prediction(word, confidence)
                st.session_state.gloss_history = st.session_state.buffer.glosses

                # Refresh right panel gloss progress bars
                with gloss_container.container():
                    for g_word, g_conf in st.session_state.gloss_history:
                        st.progress(min(1.0, max(0.0, g_conf)), text=f"**{g_word.upper()}** ({g_conf*100:.1f}%)")

            # Check if buffer has accumulated enough glosses to trigger sentence reconstruction
            if st.session_state.buffer.should_send():
                glosses_to_send = st.session_state.buffer.flush()
                sentence = st.session_state.sentence_builder.build_sentence(glosses_to_send)
                st.session_state.translated_sentence = sentence
                sentence_container.markdown(f'<div class="sentence-box">"{sentence}"</div>', unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()


if __name__ == "__main__":
    main()
