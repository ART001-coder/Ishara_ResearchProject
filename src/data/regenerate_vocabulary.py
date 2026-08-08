"""
Module: src/data/regenerate_vocabulary.py
Role: Scans data/raw/include for the words that actually exist in the
      downloaded INCLUDE categories, and writes a vocabulary.json built
      from real data instead of the hardcoded 'hospital reception' list
      (which assumed words like 'doctor'/'medicine'/'pain' that INCLUDE's
      Adjectives/Days_and_Time/Greetings/Home/People/Places/Pronouns/
      Seasons/Society categories do not actually contain).

Usage:
    python -m src.data.regenerate_vocabulary [--max_words 65]
"""
import argparse
import json
import os

from src.data.download_include import sanitize_word


def scan_available_words(include_dir: str = "data/raw/include"):
    """Returns a sorted list of (sanitized_word, category, video_count) for
    every word-folder found under include_dir."""
    entries = []
    if not os.path.exists(include_dir):
        raise FileNotFoundError(
            f"'{include_dir}' not found. Run download_include.py first."
        )

    for category in sorted(os.listdir(include_dir)):
        cat_path = os.path.join(include_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for word_folder in sorted(os.listdir(cat_path)):
            word_path = os.path.join(cat_path, word_folder)
            if not os.path.isdir(word_path):
                continue
            word = sanitize_word(word_folder)
            if not word:
                continue
            video_count = sum(
                1 for f in os.listdir(word_path)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            )
            entries.append((word, category, video_count))
    return entries


def regenerate_vocabulary(
    include_dir: str = "data/raw/include",
    vocab_file: str = "data/vocabulary.json",
    max_words: int = None,
    min_videos: int = 1,
):
    entries = scan_available_words(include_dir)

    # Drop words with too few videos to train on, then dedupe (keep first occurrence)
    seen = set()
    usable = []
    for word, category, video_count in entries:
        if video_count < min_videos or word in seen:
            continue
        seen.add(word)
        usable.append((word, category, video_count))

    if max_words is not None and len(usable) > max_words:
        usable = usable[:max_words]

    word_to_id = {word: i for i, (word, _, _) in enumerate(usable)}
    id_to_word = {str(i): word for word, i in word_to_id.items()}
    categories = {}
    for word, category, _ in usable:
        categories.setdefault(category, []).append(word)

    vocab_data = {
        "_comment": (
            f"Ishara Vocabulary: {len(word_to_id)} words auto-regenerated from real "
            f"INCLUDE folder scan (see regenerate_vocabulary.py). Replaces the original "
            f"hardcoded hospital-reception list, which did not match any downloaded "
            f"INCLUDE category (INCLUDE has no medical category)."
        ),
        "_categories_present": categories,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }

    os.makedirs(os.path.dirname(vocab_file) or ".", exist_ok=True)
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)

    print(f"[SUCCESS] Wrote {len(word_to_id)} words to '{vocab_file}'.")
    print(f"[INFO] Categories: { {k: len(v) for k, v in categories.items()} }")
    return vocab_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate vocabulary.json from real INCLUDE folders.")
    parser.add_argument("--include_dir", type=str, default="data/raw/include")
    parser.add_argument("--vocab_file", type=str, default="data/vocabulary.json")
    parser.add_argument("--max_words", type=int, default=65)
    parser.add_argument("--min_videos", type=int, default=1)
    args = parser.parse_args()

    regenerate_vocabulary(
        include_dir=args.include_dir,
        vocab_file=args.vocab_file,
        max_words=args.max_words,
        min_videos=args.min_videos,
    )
