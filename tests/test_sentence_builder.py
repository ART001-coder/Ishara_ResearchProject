"""
==============================================================================
Module: tests/test_sentence_builder.py
Role: Automated Unit Test Suite for SentenceBuilder & Offline Fallback Engine
Reference: implementation_plan.md -> Section 4.9, Section 8.3 (Bug I3), Day 9
==============================================================================
"""

import pytest
from src.inference.sentence_builder import SentenceBuilder


def test_offline_template_fallback():
    """
    Test 1: Verifies rule-based SVO template fallback works when API is absent.
    """
    builder = SentenceBuilder(api_key=None, use_fallback=True)
    glosses = [("doctor", 0.92), ("need", 0.85), ("help", 0.78)]
    result = builder.build_sentence(glosses)

    assert isinstance(result, str)
    assert len(result) > 0
    assert result.endswith(".")


def test_preloaded_cache_lookup():
    """
    Test 2: Verifies pre-cached demo sentences bypass API execution.
    """
    builder = SentenceBuilder(api_key=None, use_fallback=True)
    demo_pairs = [
        ([("stomach", 0.91), ("pain", 0.95)], "I have bad stomach pain.")
    ]
    builder.preload_cache(demo_pairs)

    result = builder.build_sentence([("stomach", 0.91), ("pain", 0.95)])
    assert result == "I have bad stomach pain."


def test_empty_gloss_handling():
    """
    Test 3: Verifies empty input gloss lists return empty string gracefully.
    """
    builder = SentenceBuilder(api_key=None, use_fallback=True)
    assert builder.build_sentence([]) == ""


if __name__ == "__main__":
    test_offline_template_fallback()
    test_preloaded_cache_lookup()
    test_empty_gloss_handling()
    print("[SUCCESS] All test_sentence_builder tests passed.")
