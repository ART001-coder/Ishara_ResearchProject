"""
==============================================================================
Module: src/inference/gloss_buffer.py
Role: Prediction Deduplication, Confidence Filtering & Sequence Buffering
Reference: implementation_plan.md -> Section 4.8, Day 3 Task 3.5 & Day 8 Task 8.2
==============================================================================
"""

import time
from typing import List, Tuple, Optional


class GlossBuffer:
    """
    Deduplication and accumulation buffer for real-time sign predictions coming from the model.
    """
    def __init__(self, conf_threshold: float = 0.5, min_glosses: int = 3, timeout_sec: float = 5.0):
        """
        Args:
            conf_threshold (float): Minimum confidence score to accept prediction (default: 0.5).
            min_glosses (int): Minimum unique glosses required before sentence building (default: 3).
            timeout_sec (float): Inactivity timeout in seconds to trigger flush (default: 5.0).
        """
        self.conf_threshold = conf_threshold
        self.min_glosses = min_glosses
        self.timeout_sec = timeout_sec
        self.glosses: List[Tuple[str, float]] = []
        self.last_word: Optional[str] = None
        self.last_add_time: float = time.time()

    def add_prediction(self, word: str, confidence: float) -> bool:
        """
        Processes a raw frame prediction, suppressing low confidence scores and consecutive repeats.
        
        Args:
            word (str): Predicted word gloss string.
            confidence (float): Softmax confidence score (0.0 to 1.0).

        Returns:
            bool: True if prediction was accepted into buffer, False if ignored.
        """
        # Skip low confidence predictions or non-sign status labels
        ignored_words = {"UNKNOWN", "NO HANDS DETECTED", "BUFFERING", "IDLE", "IDLE (RAISE HANDS TO SIGN)"}
        if confidence < self.conf_threshold or word.upper() in ignored_words:
            return False

        # Skip consecutive duplicate predictions
        if word == self.last_word:
            return False

        self.glosses.append((word, confidence))
        self.last_word = word
        self.last_add_time = time.time()
        return True

    def should_send(self) -> bool:
        """
        Checks whether accumulated glosses meet criteria to trigger LLM sentence builder.
        
        Returns:
            bool: True if >= min_glosses OR timeout reached with > 0 glosses.
        """
        if len(self.glosses) >= self.min_glosses:
            return True
        if len(self.glosses) > 0 and (time.time() - self.last_add_time) > self.timeout_sec:
            return True
        return False

    def flush(self) -> List[Tuple[str, float]]:
        """
        Flushes and returns accumulated (word, confidence) tuples, resetting buffer state.
        
        Returns:
            List[Tuple[str, float]]: List of accumulated valid (word, confidence) tuples.
        """
        result = list(self.glosses)
        self.clear()
        return result

    def clear(self) -> None:
        """Resets internal buffer state."""
        self.glosses = []
        self.last_word = None
        self.last_add_time = time.time()


if __name__ == "__main__":
    buffer = GlossBuffer(conf_threshold=0.5, min_glosses=3, timeout_sec=5.0)

    # Test deduplication
    buffer.add_prediction("doctor", 0.92)
    buffer.add_prediction("doctor", 0.90)  # Duplicate, should be skipped
    buffer.add_prediction("UNKNOWN", 0.3)  # Low conf, skipped
    buffer.add_prediction("need", 0.85)
    buffer.add_prediction("help", 0.78)

    assert buffer.should_send() is True
    flushed = buffer.flush()
    assert len(flushed) == 3
    assert [w for w, c in flushed] == ["doctor", "need", "help"]
    print("[SUCCESS] GlossBuffer verified successfully.")
