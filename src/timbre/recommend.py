from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from timbre.audio import embed_text, embed_titles
from timbre.features import extract_features_for
from timbre.ingest import read_csv_titles

DEFAULT_PHRASES = [
    "sparse melancholic folk with fingerpicked acoustic guitar",
    "hazy washed-out shoegaze with reverb-drenched guitars",
    "funky groove-driven bedroom pop with prominent bassline",
    "nostalgic synth-heavy japanese city pop",
    "bright jangly upbeat indie rock",
    "lush cinematic dream pop with airy female vocals",
    "raw lo-fi stripped-down singer-songwriter recording",
    "warm vintage jazz with brushed drums and upright bass",
]


def label_clusters(embeddings, labels, phrases=None) -> dict:
    if phrases is None:
        phrases = DEFAULT_PHRASES
    phrase_vec = embed_text(phrases)
    votes = cosine_similarity(embeddings, phrase_vec).argmax(axis=1)
    cluster_labels = {}
    for cluster_id in sorted(set(labels)):
        winning_idx = Counter(votes[labels == cluster_id]).most_common(1)[0][0]
        cluster_labels[cluster_id] = phrases[winning_idx]
    return cluster_labels


playlist_titles = read_csv_titles("data/spotify_liked_songs.csv")
titles, embeddings = embed_titles(playlist_titles, "data/liked_playlist")

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

paths = [f"data/liked_playlist/{i}.m4a" for i in range(len(titles))]
features = extract_features_for(titles, paths)

for cluster_id in sorted(set(labels)):
    cluster_titles = [
        t for t, l in zip(titles, labels) if l == cluster_id and t in features
    ]
    tempos = [features[t]["tempo"] for t in cluster_titles]
    brightness = [features[t]["brightness"] for t in cluster_titles]
    rms = [features[t]["rms"] for t in cluster_titles]
    print(f"\n[cluster {cluster_id}]  ({len(cluster_titles)} songs)")
    print(f"  tempo:      {np.mean(tempos):.1f} BPM")
    print(f"  brightness: {np.mean(brightness):.0f} Hz")
    print(f"  loudness:   {np.mean(rms):.3f}")

print("\n--- Cluster contents ---")
for cluster_id in sorted(set(labels)):
    print(f"\n[cluster {cluster_id}]")
    for label, title in zip(labels, titles):
        if label == cluster_id:
            print(f"  {title}")

centroids = kmeans.cluster_centers_

cluster_labels = label_clusters(embeddings, labels)

candidates = [
    "Daniel Caesar - Get You",
    "Bruno Major - Easily",
    "Clairo - Bags",
    "Metallica - Master of Puppets",
    "Daft Punk - Harder Better",
    "Phoebe Bridgers - Motion Sickness",
]
candidate_titles, candidate_vecs = embed_titles(candidates, "candidates")
sim_vec = cosine_similarity(candidate_vecs, centroids)

for i, title in enumerate(candidate_titles):
    best_cluster = sim_vec[i].argmax()
    best_score = sim_vec[i].max()
    label = cluster_labels[best_cluster]
    print(f"{best_score:.3f}  [{label}]  {title}")
