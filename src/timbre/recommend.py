import csv
from collections import Counter
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from timbre.audio import embed_text, embed_titles


def read_csv_titles(csv_path: str) -> list[str]:
    titles = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(f"{row['Artist Name(s)']} - {row['Track Name']}")
    return titles


playlist_titles = read_csv_titles("data/spotify_liked_songs.csv")
titles, embeddings = embed_titles(playlist_titles, "data/liked_playlist")

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

print("\n--- Cluster contents ---")
for cluster_id in sorted(set(labels)):
    print(f"\n[cluster {cluster_id}]")
    for label, title in zip(labels, titles):
        if label == cluster_id:
            print(f"  {title}")

centroids = kmeans.cluster_centers_

phrases = [
    "sparse melancholic folk with fingerpicked acoustic guitar",
    "hazy washed-out shoegaze with reverb-drenched guitars",
    "funky groove-driven bedroom pop with prominent bassline",
    "nostalgic synth-heavy japanese city pop",
    "bright jangly upbeat indie rock",
    "lush cinematic dream pop with airy female vocals",
    "raw lo-fi stripped-down singer-songwriter recording",
    "warm vintage jazz with brushed drums and upright bass",
]

phrase_vec = embed_text(phrases)

# similarity between each songs and each phrase_vec
song_phrase_sims = cosine_similarity(
    embeddings, phrase_vec
)  # (num_songs_vec, 10(phrase_vecs))

votes = song_phrase_sims.argmax(
    axis=1
)  # find the index of the most similar phrase for each song

cluster_labels = {}
for cluster_id in sorted(set(labels)):
    cluster_votes = votes[labels == cluster_id]
    winning_idx = Counter(cluster_votes).most_common(1)[0][0]
    cluster_labels[cluster_id] = phrases[winning_idx]
    tally = Counter(cluster_votes)
    print(f"\n[cluster {cluster_id}]")
    for phrase_idx, count in tally.most_common(3):
        print(f"  {count:2d} votes → {phrases[phrase_idx]}")

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

for i, title in enumerate(candidates):
    best_cluster = sim_vec[i].argmax()
    best_score = sim_vec[i].max()
    label = cluster_labels[best_cluster]
    print(f"{best_score:.3f}  [{label}]  {title}")
