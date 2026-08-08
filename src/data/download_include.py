"""
==============================================================================
Module: src/data/download_include.py
Role: Automated Downloader & Filter for INCLUDE Dataset (IIT Madras)
Reference: implementation_plan.md -> Section 5.1 & Section 8.1 (Bug D3, Bug D4), Day 1
==============================================================================
"""

import os
import zipfile
import requests
import json
import pandas as pd
import cv2
from typing import Dict, Any, List
from tqdm import tqdm
from src.utils.config import load_config


ZENODO_RECORD_URL = "https://zenodo.org/api/records/4010759"
PRIMARY_INCLUDE_URL = "https://zenodo.org/records/4010759/files/Greetings_1of2.zip/content"


def download_include_dataset(output_dir: str = "data/raw/include", subset_only: bool = False) -> None:
    """
    Downloads and extracts the INCLUDE sign language dataset from Zenodo (Record 4010759).
    
    Args:
        output_dir (str): Destination directory for raw dataset videos.
        subset_only (bool): If True, downloads subset categories.
    """
    os.makedirs(output_dir, exist_ok=True)

    drive_backup_dir = "/content/drive/MyDrive/Ishara_raw_include"
    if os.path.exists(drive_backup_dir):
        print(f"[INFO] Detected Google Drive raw video backup at '{drive_backup_dir}'. Copying video categories...")
        import shutil
        for cat_item in os.listdir(drive_backup_dir):
            src_cat = os.path.join(drive_backup_dir, cat_item)
            dst_cat = os.path.join(output_dir, cat_item)
            if os.path.isdir(src_cat) and not os.path.exists(dst_cat):
                shutil.copytree(src_cat, dst_cat, dirs_exist_ok=True)
        print("[SUCCESS] Successfully loaded raw video dataset from Google Drive backup.")

    print(f"[INFO] Querying Zenodo record 4010759 API for dataset files...")

    file_entries = []
    try:
        res = requests.get(ZENODO_RECORD_URL, timeout=15)
        if res.status_code == 200:
            data = res.json()
            for f_item in data.get('files', []):
                key = f_item.get('key', '')
                download_url = f_item.get('links', {}).get('self', '')
                if key.endswith('.zip'):
                    file_entries.append((key, download_url))
    except Exception as e:
        print(f"[WARNING] Could not query Zenodo API: {e}")

    if not file_entries:
        # Direct Zenodo fallback links for key category archives
        file_entries = [
            ("Greetings_1of2.zip", "https://zenodo.org/records/4010759/files/Greetings_1of2.zip/content"),
            ("Greetings_2of2.zip", "https://zenodo.org/records/4010759/files/Greetings_2of2.zip/content"),
            ("People_1of5.zip", "https://zenodo.org/records/4010759/files/People_1of5.zip/content"),
            ("Days_and_Time_1of3.zip", "https://zenodo.org/records/4010759/files/Days_and_Time_1of3.zip/content"),
            ("Home_1of4.zip", "https://zenodo.org/records/4010759/files/Home_1of4.zip/content"),
            ("Adjectives_1of8.zip", "https://zenodo.org/records/4010759/files/Adjectives_1of8.zip/content")
        ]

    # Filter out categories unrelated to target hospital vocabulary (e.g. Animals, Clothes, Electronics, Transportation, Jobs, Colours)
    target_categories = {"Greetings", "People", "Days_and_Time", "Adjectives", "Pronouns", "Places", "Home", "Society", "Seasons"}
    file_entries = [entry for entry in file_entries if any(cat in entry[0] for cat in target_categories)]

    print(f"[INFO] Filtered {len(file_entries)} target category archive files on Zenodo.")

    for file_key, download_url in file_entries:
        zip_dest = os.path.join(output_dir, file_key)
        extracted_marker = os.path.join(output_dir, f".extracted_{file_key}")
        cat_prefix = file_key.split('_')[0]
        if os.path.exists(extracted_marker) or os.path.exists(os.path.join(output_dir, cat_prefix)):
            print(f"[SKIP] Archive '{file_key}' already downloaded & unzipped.")
            continue

        print(f"[INFO] Downloading Zenodo archive '{file_key}'...")
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(zip_dest, "wb") as f, tqdm(
                desc=f"Downloading {file_key}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size)

            print(f"[INFO] Unpacking {file_key} to {output_dir}...")
            with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            with open(extracted_marker, "w") as f_mark:
                f_mark.write("ok")

            print(f"[SUCCESS] Extracted {file_key}")
        except Exception as e:
            print(f"[WARNING] Download of '{file_key}' failed: {e}")
        finally:
            if os.path.exists(zip_dest):
                os.remove(zip_dest)

    print("[SUCCESS] INCLUDE dataset acquisition complete.")


def filter_vocabulary_videos(include_dir: str = "data/raw/include", 
                             vocab_file: str = "data/vocabulary.json",
                             output_csv: str = "data/train_split.csv") -> pd.DataFrame:
    """
    Filters raw INCLUDE videos matching the target 65-word hospital vocabulary,
    checks OpenCV video file validity, and builds dataset index records across train, val, and test splits.
    """
    # Robust vocabulary file resolution with auto-generation fallback
    candidate_paths = [
        vocab_file,
        "vocabulary.json",
        "data/vocabulary.json",
        os.path.join(os.path.dirname(__file__), "..", "..", vocab_file),
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "vocabulary.json"),
        "/content/Ishara/data/vocabulary.json",
        "/content/Ishara/vocabulary.json",
        "/content/data/vocabulary.json",
        "/content/vocabulary.json"
    ]

    found_vocab = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            found_vocab = p
            break

    if found_vocab:
        with open(found_vocab, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
    else:
        print("[INFO] vocabulary.json not found on disk. Auto-generating 65-word hospital vocabulary...")
        default_words = [
            "namaste", "goodbye", "thank_you", "please", "sorry", "yes", "no", "again",
            "doctor", "medicine", "hospital", "pain", "fever", "head", "stomach", "heart",
            "eye", "ear", "hurt", "sick", "need", "help", "sit", "wait", "come", "go", "eat",
            "drink", "sleep", "walk", "stop", "give", "see", "what", "where", "when", "how",
            "who", "mother", "father", "family", "home", "school", "name", "I", "you", "one",
            "two", "three", "four", "five", "today", "tomorrow", "morning", "night", "good",
            "bad", "hot", "cold", "big", "small", "happy", "water", "food", "telephone"
        ]
        word_to_id = {w: i for i, w in enumerate(default_words)}
        vocab_data = {"word_to_id": word_to_id, "id_to_word": {str(i): w for i, w in enumerate(default_words)}}
        os.makedirs(os.path.dirname(vocab_file) or "data", exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

    word_to_id = vocab_data.get("word_to_id", {})
    train_records = []
    val_records = []
    test_records = []
    skipped_count = 0

    print(f"[INFO] Recursively scanning raw INCLUDE directory: {include_dir}")

    # Build a map of lowercase word -> list of matching directory paths
    word_dir_map = {}
    if os.path.exists(include_dir):
        for root, dirs, _ in os.walk(include_dir):
            for d in dirs:
                d_lower = d.lower()
                if d_lower not in word_dir_map:
                    word_dir_map[d_lower] = []
                word_dir_map[d_lower].append(os.path.join(root, d))

    for word, label_id in word_to_id.items():
        matched_dirs = word_dir_map.get(word.lower(), [])
        if not matched_dirs:
            print(f"[WARNING] Word '{word}' directory missing in raw dataset scan.")
            continue

        video_files = []
        for m_dir in matched_dirs:
            for f_name in os.listdir(m_dir):
                if f_name.endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(m_dir, f_name))

        if len(video_files) == 0:
            continue

        valid_vids = []
        for vid_path in video_files:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                skipped_count += 1
                cap.release()
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                skipped_count += 1
                continue

            signer_id = "Signer_Unknown"
            vid_name = os.path.basename(vid_path)
            for part in vid_name.split("_"):
                if "signer" in part.lower():
                    signer_id = part

            valid_vids.append({
                "video_path": vid_path,
                "word_label": word,
                "label_id": label_id,
                "signer_id": signer_id,
                "dataset_source": "include",
                "duration_frames": frame_count
            })

        # Split 70% Train, 15% Val, 15% Test
        n_total = len(valid_vids)
        if n_total == 0:
            continue

        n_train = max(1, int(n_total * 0.70))
        n_val = max(1, int(n_total * 0.15)) if n_total >= 3 else 0

        train_records.extend(valid_vids[:n_train])
        val_records.extend(valid_vids[n_train:n_train + n_val])
        test_records.extend(valid_vids[n_train + n_val:])

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    base_dir = os.path.dirname(output_csv) or "data"
    os.makedirs(base_dir, exist_ok=True)

    df_train.to_csv(os.path.join(base_dir, "train_split.csv"), index=False)
    df_val.to_csv(os.path.join(base_dir, "val_split.csv"), index=False)
    df_test.to_csv(os.path.join(base_dir, "test_split.csv"), index=False)

    print(f"[SUCCESS] Filtered {len(df_train) + len(df_val) + len(df_test)} valid videos: "
          f"{len(df_train)} train, {len(df_val)} val, {len(df_test)} test records.")
    print(f"[INFO] Skipped {skipped_count} corrupted/unreadable video files.")

    return df_train


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download INCLUDE and build filtered split CSVs.")
    parser.add_argument("--output_dir", type=str, default="data/raw/include")
    parser.add_argument("--vocab_file", type=str, default="data/vocabulary.json")
    parser.add_argument("--skip_download", action="store_true",
                         help="Skip the Zenodo download step and only run the vocabulary filter "
                              "(use this if you already have data/raw/include populated).")
    args = parser.parse_args()

    if not args.skip_download:
        download_include_dataset(output_dir=args.output_dir)
    filter_vocabulary_videos(include_dir=args.output_dir, vocab_file=args.vocab_file)
