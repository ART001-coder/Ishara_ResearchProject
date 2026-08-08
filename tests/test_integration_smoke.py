"""
Module: tests/test_integration_smoke.py
Role: End-to-end smoke test that threads Stage 4 (classifier) -> Stage 5
      (gloss buffer + sentence builder) together using synthetic data.

This does NOT require: a GPU, the INCLUDE dataset, MediaPipe, a webcam, or a
Gemini API key. It exists to catch integration-breaking bugs (e.g. an
entrypoint that's defined but never called) that per-module unit tests miss.
"""
import json
import os
import random
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.model.lstm_model import ISLRecognizer
from src.model.train import run_training
from src.inference.gloss_buffer import GlossBuffer
from src.inference.sentence_builder import SentenceBuilder


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    """Builds a tiny 5-class synthetic dataset + config.yaml under tmp_path
    and chdirs into it, mirroring the real project's directory layout."""
    monkeypatch.chdir(tmp_path)

    words = [f"word{i}" for i in range(5)]
    word_to_id = {w: i for i, w in enumerate(words)}
    os.makedirs("data", exist_ok=True)
    json.dump(
        {"word_to_id": word_to_id, "id_to_word": {str(v): k for k, v in word_to_id.items()}},
        open("data/vocabulary.json", "w"),
    )

    rows = {"train": [], "val": [], "test": []}
    for w in words:
        for split, n in [("train", 6), ("val", 2), ("test", 2)]:
            d = f"data/processed/{split}/{w}"
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                vid = f"{w}_{split}_{i}"
                length = random.randint(15, 35)
                np.save(
                    os.path.join(d, f"{vid}.npy"),
                    (np.random.randn(length, 225) * 0.1 + word_to_id[w]).astype("float32"),
                )
                rows[split].append({"video_path": vid + ".mp4", "word_label": w, "label_id": word_to_id[w]})

    for split in rows:
        pd.DataFrame(rows[split]).to_csv(f"data/{split}_split.csv", index=False)

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
    cfg["model"]["num_classes"] = 5
    cfg["project"]["target_vocab_size"] = 5
    cfg["training"]["num_epochs"] = 2
    cfg["training"]["batch_size"] = 4
    cfg["training"]["use_class_weights"] = False
    yaml.safe_dump(cfg, open("config.yaml", "w"))

    return words, word_to_id


def test_training_actually_runs_and_saves_checkpoint(synthetic_project):
    """Regression test: run_training() must be reachable from the CLI entrypoint
    and must produce a checkpoint file. (Previously the __main__ block only
    printed a message and never called run_training().)"""
    run_training("config.yaml")
    assert os.path.exists("checkpoints/best_model.pth")


def test_full_pipeline_model_to_sentence(synthetic_project):
    """Stage 4 -> 5 -> 6 (minus UI): trained model -> gloss buffer -> sentence builder."""
    words, word_to_id = synthetic_project
    run_training("config.yaml")

    id_to_word = {str(v): k for k, v in word_to_id.items()}
    model = ISLRecognizer(num_classes=5)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
    model.eval()

    gloss_buffer = GlossBuffer(conf_threshold=0.1, min_glosses=2, timeout_sec=5.0)
    sentence_builder = SentenceBuilder(api_key=None, use_fallback=True)

    for cls in [0, 1, 2]:
        x = torch.randn(1, 30, 225) * 0.1 + cls
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
        gloss_buffer.add_prediction(id_to_word[str(pred.item())], conf.item())

    assert gloss_buffer.should_send()
    glosses = gloss_buffer.flush()
    assert len(glosses) >= 2

    sentence = sentence_builder.build_sentence(glosses)
    assert isinstance(sentence, str) and len(sentence) > 0
