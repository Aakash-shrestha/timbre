from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from timbre.audio import embed_playlist, embed_titles

PLAYLIST_ID = "PLbAAt7n1yO_tydN0yogQXwvjH-KpkoZha"


titles, embeddings = embed_playlist(playlist_id=PLAYLIST_ID)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

candidates = [
    "Daniel Caesar - Get You",
    "Bruno Major - Easily",
    "Clairo - Bags",
    "Metallica - Master of Puppets",
    "Daft Punk - Harder Better",
    "Phoebe Bridgers - Motion Sickness",
]

outdir = "candidates"
Path(outdir).mkdir(exist_ok=True)
candidate_titles, candidtate_vecs = embed_titles(candidates, outdir)

centroids = kmeans.cluster_centers_

sim_vec = cosine_similarity(candidtate_vecs, centroids)

for i, title in enumerate(candidate_titles):
    best_cluster = sim_vec[i].argmax()
    best_score = sim_vec[i].max()
    print(f"{best_score:.3f}  [cluster {best_cluster}]  {title}")
