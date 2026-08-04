"""
==============================================================================
Module: src/inference/sentence_builder.py
Role: LLM API Sentence Reconstruction & Rule-Based Template Fallback Engine
Reference: implementation_plan.md -> Section 1.5, Section 4.9, Section 8.3 (Bug I3), Day 3/7
==============================================================================
"""

import os
import time
from typing import List, Tuple, Dict


class SentenceBuilder:
    """
    Reconstructs raw sign glosses into grammatically correct sentences using Gemini 2.0 Flash or Fallback.
    """
    SYSTEM_PROMPT = """You are an Indian Sign Language (ISL) translator.

You will receive a sequence of English words (glosses) detected from ISL signing,
along with a confidence score (0.0 to 1.0) for each word.

Rules:
1. Reconstruct into ONE grammatically correct English sentence.
2. Confidence >= 0.7 = HIGH — preserve exactly.
3. Confidence < 0.7 = LOW — may reinterpret if context helps.
4. DO NOT invent new information.
5. You may add articles, prepositions, tense markers.
6. Context: hospital/reception scenario.
7. Output ONLY the final sentence.

Examples:
Input:  [doctor (0.92), need (0.85), help (0.78)]
Output: I need help from the doctor.

Input:  [stomach (0.91), pain (0.95), bad (0.80)]
Output: I have bad stomach pain.

Input:  [hello (0.99), name (0.88), what (0.75)]
Output: Hello, what is your name?
"""

    def __init__(self, api_key: str = None, timeout: float = 3.0, use_fallback: bool = True):
        """
        Args:
            api_key (str): Google Gemini API key string.
            timeout (float): Timeout seconds for API call before falling back.
            use_fallback (bool): Whether to enable rule-based template fallback on API failure.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        self.use_fallback = use_fallback
        self.cache: Dict[Tuple[str, ...], str] = {}
        self.model = None

        if self.api_key:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel("gemini-2.0-flash")
            except Exception as e:
                print(f"[WARNING] Could not initialize Gemini API client: {e}")

    def build_sentence(self, glosses: List[Tuple[str, float]]) -> str:
        """
        Converts a list of (word, confidence) glosses into a grammatically correct sentence string.
        
        Args:
            glosses (list): List of (word_string, confidence_float) tuples.
            
        Returns:
            str: Natural English sentence string.
        """
        if not glosses:
            return ""

        # Step 1: Check Pre-Cached Demo Sentences (Zero Latency - Section 11.2 R6)
        cache_key = tuple(w.lower() for w, c in glosses)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Step 2: Format prompt for Gemini API
        gloss_str = ", ".join(f"{w} ({c:.2f})" for w, c in glosses)
        user_prompt = f"Glosses: [{gloss_str}]"

        # Step 3: Attempt Gemini API Call if client initialized
        if self.model:
            try:
                response = self._call_gemini_api(user_prompt)
                if response:
                    return response
            except Exception as e:
                print(f"[WARNING] Gemini API call failed ({e}). Falling back to rule-based template (Bug I3).")

        # Step 4: Fallback to rule-based sentence builder
        if self.use_fallback:
            return self._template_fallback(glosses)

        return " ".join(w for w, c in glosses)

    def _call_gemini_api(self, user_prompt: str) -> str:
        """Executes content generation request to Gemini model."""
        response = self.model.generate_content(
            [self.SYSTEM_PROMPT, user_prompt],
            generation_config={"temperature": 0.3, "max_output_tokens": 100}
        )
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        return ""

    def _template_fallback(self, glosses: List[Tuple[str, float]]) -> str:
        """
        Rule-based heuristic sentence builder for offline / API-down scenarios.
        Reorders words into Subject-Verb-Object format for all 65 hospital vocabulary terms.
        """
        raw_words = [w for w, c in glosses]
        if not raw_words:
            return ""

        words_lower = [w.lower() for w in raw_words]

        subjects_set = {"i", "you", "mother", "father", "family", "doctor", "who"}
        verbs_set = {"help", "sit", "wait", "come", "go", "eat", "drink", "sleep", "walk", "stop", "give", "see", "need", "hurt"}
        medical_set = {"doctor", "medicine", "hospital", "pain", "fever", "head", "stomach", "heart", "eye", "ear", "hurt", "sick"}
        questions_set = {"what", "where", "when", "how", "who"}

        subj = [w for w, wl in zip(raw_words, words_lower) if wl in subjects_set]
        verb = [w for w, wl in zip(raw_words, words_lower) if wl in verbs_set]
        question = [w for w, wl in zip(raw_words, words_lower) if wl in questions_set]
        rest = [w for w, wl in zip(raw_words, words_lower) if wl not in subjects_set and wl not in verbs_set and wl not in questions_set]

        # Handle Question Patterns
        if question:
            q_word = question[0].capitalize()
            rest_clean = " ".join(w.replace("_", " ") for w in rest + subj if w.lower() not in questions_set)
            if rest_clean:
                return f"{q_word} is {rest_clean}?"
            return f"{q_word}?"

        # Handle Medical Symptom Patterns (e.g. stomach pain)
        medical_present = [w for w, wl in zip(raw_words, words_lower) if wl in medical_set]
        if medical_present and not verb:
            clean_med = " ".join(w.replace("_", " ") for w in rest)
            lead_subj = subj[0] if subj else "I"
            return f"{lead_subj.capitalize()} have {clean_med}."

        # Standard SVO Assembly
        if not subj and verb:
            subj = ["I"]

        sentence_words = [w.replace("_", " ") for w in (subj + verb + rest)]
        sentence = " ".join(sentence_words).capitalize() + "."
        return sentence

    def preload_cache(self, demo_sentences: List[Tuple[List[Tuple[str, float]], str]]) -> None:
        """
        Pre-caches demo sentences for instant presentation response.
        
        Args:
            demo_sentences (list): List of (gloss_tuple_list, sentence_string) pairs.
        """
        for gloss_list, sentence in demo_sentences:
            key = tuple(w for w, c in gloss_list)
            self.cache[key] = sentence


if __name__ == "__main__":
    builder = SentenceBuilder(api_key=None, use_fallback=True)
    sample_glosses = [("doctor", 0.92), ("need", 0.85), ("help", 0.78)]
    sentence = builder.build_sentence(sample_glosses)
    assert len(sentence) > 0
    assert sentence.endswith(".")
    print(f"[SUCCESS] SentenceBuilder fallback test passed: '{sentence}'")

    # Test preloading cache
    builder.preload_cache([
        ([("stomach", 0.91), ("pain", 0.95)], "I have bad stomach pain.")
    ])
    cached_sentence = builder.build_sentence([("stomach", 0.91), ("pain", 0.95)])
    assert cached_sentence == "I have bad stomach pain."
    print(f"[SUCCESS] SentenceBuilder cache test passed: '{cached_sentence}'")
