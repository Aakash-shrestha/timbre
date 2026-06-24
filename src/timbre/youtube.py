import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["YOUTUBE_API_KEY"]
PLAYLIST_ID = "PLbAAt7n1yO_tydN0yogQXwvjH-KpkoZha"


def filter_songs_title(title: str) -> bool:
    blacklist_keywords = [
        "lesson",
        "tutorial",
        "masterclass",
        "how to",
        "chords",
        "playlist",
        "mix",
        "lofi remix",
        "songs that",
        "full album",
        "unboxing",
        "detox",
        "routine",
        "private video",
    ]

    for keyword in blacklist_keywords:
        if keyword in title.lower():
            return False
    return True


def get_titles(playlist_id: str):
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        params={
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": API_KEY,
        },
    )
    data = response.json()
    titles = [item["snippet"]["title"] for item in data["items"]]
    return [t for t in titles if filter_songs_title(t)]

