import pickle
import time
from pathlib import Path

import laion_clap
import numpy as np
import requests

PLAYLIST_ID = "PLbAAt7n1yO_tydN0yogQXwvjH-KpkoZha"

CACHE_PATH = "embedding_cache.pkl"


def load_cache() -> dict:
    if Path(CACHE_PATH).exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_text(phrases: list[str]) -> np.ndarray:
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model.get_text_embedding(phrases, use_tensor=False)


ITUNES_THROTTLE_SECONDS = 0.5
ITUNES_MAX_RETRIES = 3


def search_itunes(term: str) -> str | None:
    """
    Search for the given term on the iTunes API and return the preview URL of the first result.
    """
    for attempt in range(ITUNES_MAX_RETRIES):
        time.sleep(ITUNES_THROTTLE_SECONDS)
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": term, "entity": "song", "limit": 1},
        )
        if response.status_code == 403:
            time.sleep(2**attempt)
            continue
        if not response.text.strip():
            return None
        try:
            results = response.json()["results"]
        except requests.exceptions.JSONDecodeError:
            return None
        if not results:
            return None
        return results[0]["previewUrl"]
    return None


def download_preview(url: str, path: str):
    """
    Download the preview from the given URL and save it to the specified path.
    """
    resp = requests.get(url)
    with open(path, "wb") as f:
        f.write(resp.content)


_model = None


def _get_model() -> laion_clap.CLAP_Module:
    global _model
    if _model is None:
        _model = laion_clap.CLAP_Module(enable_fusion=False)
        _model.load_ckpt()
    return _model


def _embed_batch(paths: list[str]) -> np.ndarray:
    return _get_model().get_audio_embedding_from_filelist(x=paths, use_tensor=False)


BATCH_SIZE = 16


def embed_titles(
    titles: list[str], outdir: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    cache = load_cache()
    resolved_titles = []
    vectors = []
    misses = []
    miss_paths = []
    urls = []
    miss_urls = []

    Path(outdir).mkdir(exist_ok=True)
    for title in titles:
        if title in cache:
            resolved_titles.append(title)
            vectors.append(cache[title]["embedding"])
            urls.append(cache[title]["preview_url"])
            continue

        url = search_itunes(title)
        if url is None:
            print(f"No preview found, skipping: {title}")
            continue

        current_path = f"{outdir}/{len(miss_paths)}.m4a"
        if not Path(current_path).exists():
            download_preview(url, current_path)
        misses.append(title)
        miss_paths.append(current_path)
        miss_urls.append(url)

    if miss_paths:
        new_vecs = []
        for i in range(0, len(miss_paths), BATCH_SIZE):
            batch = miss_paths[i : i + BATCH_SIZE]
            new_vecs.extend(_embed_batch(batch))

        for title, vec, url in zip(misses, new_vecs, miss_urls):
            cache[title] = {"embedding": vec, "preview_url": url}
            resolved_titles.append(title)
            vectors.append(vec)
            urls.append(url)
        save_cache(cache)

    return resolved_titles, np.array(vectors), urls


def embed_playlist(playlist_id: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    from timbre.youtube import get_titles

    titles = get_titles(playlist_id)
    return embed_titles(titles, "previews")
