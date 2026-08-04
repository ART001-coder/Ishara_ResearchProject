"""
==============================================================================
Module: src/utils/visualize.py
Role: Keypoint Skeleton & UI Overlay Rendering Helper
Reference: implementation_plan.md -> Section 4.4 & Section 4.10
==============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp


def draw_landmarks_on_frame(frame_rgb: np.ndarray, results) -> np.ndarray:
    """
    Renders MediaPipe pose and hand keypoint connections on RGB image frame.
    
    Args:
        frame_rgb (np.ndarray): Input RGB frame (H, W, 3).
        results: MediaPipe Holistic process results object.
        
    Returns:
        np.ndarray: Annotated RGB frame.
    """
    if results is None:
        return frame_rgb

    annotated = frame_rgb.copy()

    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        # RGB Color specifications for MediaPipe drawing
        pose_spec = mp_drawing.DrawingSpec(color=(40, 200, 100), thickness=2, circle_radius=3)    # Emerald Green
        hand_spec = mp_drawing.DrawingSpec(color=(235, 80, 140), thickness=2, circle_radius=3)   # Soft Rose
        conn_spec = mp_drawing.DrawingSpec(color=(255, 215, 0), thickness=1, circle_radius=1)    # Gold Accent

        # Pose landmarks
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_spec,
                connection_drawing_spec=conn_spec
            )

        # Left hand landmarks
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_spec,
                connection_drawing_spec=conn_spec
            )

        # Right hand landmarks
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_spec,
                connection_drawing_spec=conn_spec
            )
    except Exception:
        pass

    return annotated


def render_prediction_overlay(frame_rgb: np.ndarray, word: str, confidence: float) -> np.ndarray:
    """
    Draws a sleek semi-transparent top banner and confidence indicator on the RGB image frame.
    
    Args:
        frame_rgb (np.ndarray): Input RGB image frame.
        word (str): Predicted sign word label string.
        confidence (float): Confidence score between 0.0 and 1.0.
        
    Returns:
        np.ndarray: RGB Image frame with prediction banner overlay.
    """
    annotated = frame_rgb.copy()
    h, w, _ = annotated.shape

    # RGB Color palette based on prediction confidence
    if confidence >= 0.7:
        banner_color = (25, 135, 84)    # Emerald Green (RGB)
        bar_color = (40, 220, 130)
    elif confidence >= 0.5:
        banner_color = (217, 119, 6)    # Amber/Gold (RGB)
        bar_color = (251, 191, 36)
    else:
        banner_color = (71, 85, 105)    # Sleek Slate Gray (RGB)
        bar_color = (148, 163, 184)

    # Create semi-transparent banner overlay (top 10% of frame height)
    banner_height = max(45, int(h * 0.10))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)

    # Alpha blend banner for a polished semi-transparent look
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    # Format text label
    text_str = f"Sign: {word.upper()} ({confidence * 100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, w / 900.0)
    thickness = max(1, int(w / 450.0))

    # Text positioning with slight shadow for high contrast readability
    text_x = 18
    text_y = int(banner_height * 0.65)
    cv2.putText(annotated, text_str, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, text_str, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Bottom progress bar indicator
    bar_width = int(w * min(1.0, max(0.0, confidence)))
    cv2.rectangle(annotated, (0, banner_height - 5), (bar_width, banner_height), bar_color, -1)

    return annotated


if __name__ == "__main__":
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = render_prediction_overlay(dummy_frame, "doctor", 0.92)
    assert out.shape == (480, 640, 3)
    print("[SUCCESS] src/utils/visualize.py tested successfully.")
