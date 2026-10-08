"""
Pulls BPM, key, and a rough energy level out of an audio file. Beat tracking
and chroma extraction are librosa's job (reimplementing an onset detector
badly would help no one). The key-finding and anything mixing-related is
this project's own.

Analysis is slow-ish (a few seconds per track), so results get cached to
disk next to wherever you're running this from, keyed on the file's path +
size + mtime. Re-run on a crate you've already analyzed and it's instant.
"""

import hashlib
import json
import os

import librosa
import numpy as np

from keyfinder import estimate_key
from camelot import to_camelot, key_name

CACHE_DIR = ".setlist_cache"
SR = 22050  # downsample for speed, plenty for beat/chroma analysis


def _cache_key(path):
    stat = os.stat(path)
    raw = f"{path}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(raw.encode()).hexdigest()


def analyze_track(path, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, _cache_key(path) + ".json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    y, sr = librosa.load(path, sr=SR, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pitch_class, is_minor, confidence = estimate_key(chroma.mean(axis=1))

    rms = float(librosa.feature.rms(y=y).mean())

    result = {
        "path": path,
        "title": os.path.splitext(os.path.basename(path))[0],
        "bpm": round(bpm, 1),
        "pitch_class": pitch_class,
        "is_minor": bool(is_minor),
        "key": key_name(pitch_class, is_minor),
        "camelot": to_camelot(pitch_class, is_minor),
        "key_confidence": round(confidence, 3),
        "energy": round(rms, 5),
        "duration_sec": round(len(y) / sr, 1),
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def analyze_folder(folder, extensions=(".mp3", ".wav", ".m4a", ".flac"), use_cache=True):
    tracks = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(extensions):
            path = os.path.join(folder, name)
            try:
                tracks.append(analyze_track(path, use_cache=use_cache))
            except Exception as e:
                print(f"skipping {name}: {e}")

    if tracks:
        energies = [t["energy"] for t in tracks]
        lo, hi = min(energies), max(energies)
        span = hi - lo or 1.0
        for t in tracks:
            t["energy_norm"] = round((t["energy"] - lo) / span, 3)

    return tracks


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    for t in analyze_folder(folder):
        print(f"{t['title']:40s}  {t['bpm']:6.1f} bpm  {t['camelot']:>3s} ({t['key']:3s})  energy={t['energy_norm']:.2f}")
