import json

import umap
from sklearn.cluster import KMeans

from timbre.audio import embed_titles
from timbre.ingest import read_csv_titles

playlist_titles = read_csv_titles("data/spotify_liked_songs.csv")
title, embeddings = embed_titles(playlist_titles, "data/liked_playlist")
labels = KMeans(n_clusters=3, random_state=42).fit_predict(embeddings)


# dimensionality reduction using UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords_2d = reducer.fit_transform(embeddings)
print(coords_2d.shape)  # (190, 2)

data = [
    {"title": title, "x": float(x), "y": float(y), "cluster": int(cluster)}
    for title, (x, y), cluster in zip(title, coords_2d, labels)
]

with open("data/taste_map.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Exported {len(data)} points to data/taste_map.json")
