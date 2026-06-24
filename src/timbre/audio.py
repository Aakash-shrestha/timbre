from pathlib import Path

import laion_clap
import numpy as np
import requests

from timbre.youtube import get_titles

PLAYLIST_ID = "PLbAAt7n1yO_tydN0yogQXwvjH-KpkoZha"


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
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    resolved_titles = []
    path = []

    Path(outdir).mkdir(exist_ok=True)
    for title in titles:
        url = search_itunes(title)

        if url is None:
            print(f"No preview found, skipping: {title}")
            continue

        current_path = f"{outdir}/{len(path)}.m4a"
        if not Path(current_path).exists():
            download_preview(url, current_path)
        resolved_titles.append(title)
        path.append(current_path)

    embedding = model.get_audio_embedding_from_filelist(x=path, use_tensor=False)

    return resolved_titles, embedding


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
