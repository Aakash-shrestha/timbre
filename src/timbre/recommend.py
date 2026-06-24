import csv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from timbre.audio import embed_titles


def read_csv_titles(csv_path: str) -> list[str]:
    titles = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(f"{row['Artist Name(s)']} - {row['Track Name']}")
    return titles


playlist_titles = read_csv_titles("data/spotify_liked_songs.csv")
titles, embeddings = embed_titles(playlist_titles, "data/liked_playlist")

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")
    print(f"k={k}: silhouette={score:.4f}")

print("\n--- Cluster contents ---")
for cluster_id in sorted(set(labels)):
    print(f"\n[cluster {cluster_id}]")
    for label, title in zip(labels, titles):
        if label == cluster_id:
            print(f"  {title}")

candidates = [
    "Daniel Caesar - Get You",
    "Bruno Major - Easily",
    "Clairo - Bags",
    "Metallica - Master of Puppets",
    "Daft Punk - Harder Better",
    "Phoebe Bridgers - Motion Sickness",
]

liked_set = set(playlist_titles)
overlapping = [c for c in candidates if c in liked_set]
if overlapping:
    print(f"\nWarning: candidates overlap with liked songs: {overlapping}")

outdir = "candidates"
Path(outdir).mkdir(exist_ok=True)
candidate_titles, candidate_vecs = embed_titles(candidates, outdir)
print(f"candidates resolved: {candidate_titles}")

centroids = kmeans.cluster_centers_

sim_vec = cosine_similarity(candidate_vecs, centroids)

print("\n--- Candidate scores ---")
for i, title in enumerate(candidate_titles):
    best_cluster = sim_vec[i].argmax()
    best_score = sim_vec[i].max()
    print(f"{best_score:.3f}  [cluster {best_cluster}]  {title}")
