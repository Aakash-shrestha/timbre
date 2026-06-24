from sklearn.cluster import KMeans

from timbre.audio import embed_playlist

PLAYLIST_ID = "PLbAAt7n1yO_tydN0yogQXwvjH-KpkoZha"

titles, embeddings = embed_playlist(playlist_id=PLAYLIST_ID)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)
for label, title in sorted(zip(labels, titles)):
    print(f"[cluster {label}] {title}")
