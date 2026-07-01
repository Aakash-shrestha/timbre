import pickle
import warnings
from pathlib import Path

import librosa
import numpy as np

from timbre.audio import embed_titles
from timbre.io import read_csv_titles

FEATURES_CACHE = "features_cache.pkl"


def load_features_cache() -> dict:
    if Path(FEATURES_CACHE).exists():
        with open(FEATURES_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save_features_cache(cache: dict):
    with open(FEATURES_CACHE, "wb") as f:
        pickle.dump(cache, f)


def extract_features_for(titles: list[str], paths: list[str]) -> dict:
    cache = load_features_cache()
    computed_any = False

    for title, path in zip(titles, paths):
        if title in cache:
            continue
        if not Path(path).exists():
            print(f"No audio file, skipping features: {title}")
            continue
        feature = extract_features(path)
        cache[title] = feature
        computed_any = True
    if computed_any:
        save_features_cache(cache)

    return {t: cache[t] for t in titles if t in cache}


def extract_features(path: str) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings(
            "ignore", message=".*audioread.*", category=FutureWarning
        )
        y, sr = librosa.load(path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    return {"tempo": float(tempo[0]), "brightness": float(centroid), "rms": float(rms)}
