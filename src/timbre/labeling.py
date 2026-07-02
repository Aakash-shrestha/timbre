from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from timbre.audio import embed_text

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
    for cluster_id in set(labels):
        winning_idx = Counter(votes[labels == cluster_id]).most_common(1)[0][0]
        cluster_labels[cluster_id] = phrases[winning_idx]
    return cluster_labels
