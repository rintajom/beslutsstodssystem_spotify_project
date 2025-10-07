import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from contentBasedRecommender import ContentBasedRecommender


#print(df.describe())
# EDA to do later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_spotify_data.csv")

    recommender = ContentBasedRecommender(df)
    recommender.fit()

    test_song = "MMS"

    print(f'\nContent-Based Recommendations:\n')
    print("Track info:")
    print("-" * 50)
    track_info = recommender.track_info(test_song)
    print(f'Track: {track_info["track_name"]}')
    print(f'Artist: {track_info["track_artist"]}')
    print(f'Album: {track_info["track_album_name"]} ({track_info["track_album_release_date"]})')
    print(f'Popularity: {track_info["track_popularity"]}')
    print(f'Tempo: {track_info["tempo"]} BPM')
    print(f'Loudness: {track_info["loudness"]} dB')
    print(f'Danceability: {track_info["danceability"]}')

    print("\nRecommended Tracks")
    print("-" * 50)
    recommendations = recommender.recommend(test_song, num_recs=5)

    # TODO: gör det här till en function
    for i, rec in enumerate(recommendations, start=1):
        track_name = rec.get('track_name', 'Unknown')
        artist = rec.get('track_artist', 'Unknown')
        popularity = rec.get('track_popularity', 'N/A')
        similarity = rec.get('similarity_score', 0)
        print(f"{i}.{track_name} — {artist}")
        print(f"Popularity: {popularity}   |   Similarity: {similarity:.2f}")
        print("-" * 50)
