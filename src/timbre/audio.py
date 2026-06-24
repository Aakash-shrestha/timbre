import pickle
from pathlib import Path

import laion_clap
import numpy as np
import requests

from timbre.youtube import get_titles

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


def search_itunes(term: str) -> str | None:
    """
    Search for the given term on the iTunes API and return the preview URL of the first result.
    """
    response = requests.get(
        "https://itunes.apple.com/search",
        params={"term": term, "entity": "song", "limit": 1},
    )
    results = response.json()["results"]
    if not results:
        return None
    return results[0]["previewUrl"]


def download_preview(url: str, path: str):
    """
    Download the preview from the given URL and save it to the specified path.
    """
    resp = requests.get(url)
    with open(path, "wb") as f:
        f.write(resp.content)


def embed_titles(titles: list[str], outdir: str) -> tuple[list[str], np.ndarray]:
    cache = load_cache()
    model = None  # lazy load the model if there are any misses
    resolved_titles = []
    vectors = []
    misses = []
    miss_paths = []

    Path(outdir).mkdir(exist_ok=True)
    for title in titles:
        if title in cache:
            resolved_titles.append(title)
            vectors.append(cache[title])
            continue

        url = search_itunes(title)
        if url is None:
            print(f"No preview found, skipping: {title}")
            continue

        current_path = f"{outdir}/{len(miss_paths)}.m4a"
        misses.append(title)
        miss_paths.append(current_path)
    if miss_paths:
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        new_vecs = model.get_audio_embedding_from_filelist(
            x=miss_paths, use_tensor=False
        )

        for title, vecs in zip(misses, new_vecs):
            cache[title] = vecs
            resolved_titles.append(title)
            vectors.append(vecs)
        save_cache(cache)

    return resolved_titles, np.array(vectors)


def embed_playlist(playlist_id: str) -> tuple[list[str], np.ndarray]:
    """
    Given a YouTube playlist ID, this function retrieves the titles of the videos in the playlist,
    searches for their previews on the iTunes API, downloads the previews, and returns a list of the resolved
    titles and their corresponding audio embeddings.
    """
    titles = get_titles(playlist_id)
    return embed_titles(titles, "previews")


resolved_titles, embeddings = embed_playlist(PLAYLIST_ID)
for title, embedding in zip(resolved_titles, embeddings):
    print(f"Title: {title}, Embedding shape: {embedding.shape}")

print("embedding shape: ", embeddings.shape)
