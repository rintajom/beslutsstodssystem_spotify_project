import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from contentBasedRecommender import ContentBasedRecommender
from ruleBasedRecommender import RuleBasedRecommender


#print(df.describe())
# EDA to do later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_spotify_data.csv")
    content_recommender = ContentBasedRecommender(df)
    rule_recommender = RuleBasedRecommender(df)

    test_song = "Espresso"

    def print_recommendations(recommendations):
        for i, rec in enumerate(recommendations, start=1):
            track_name = rec.get('track_name', 'Unknown')
            artist = rec.get('track_artist', 'Unknown')
            popularity = rec.get('track_popularity', 'N/A')
            similarity = rec.get('similarity_score', 0)
            print(f"{i}.{track_name} — {artist}")
            print(f"Popularity: {popularity}   |   Similarity: {similarity:.2f}")
            print("-" * 50)

    content_recommender.fit()

    print("\nTrack info:")
    print("-" * 50)
    track_info = content_recommender.track_info(test_song)
    print(f'Track: {track_info["track_name"]}')
    print(f'Artist: {track_info["track_artist"]}')
    print(f'Album: {track_info["track_album_name"]} ({track_info["track_album_release_date"]})')
    print(f'Popularity: {track_info["track_popularity"]}')
    print(f'Tempo: {track_info["tempo"]} BPM')
    print(f'Loudness: {track_info["loudness"]} dB')
    print(f'Danceability: {track_info["danceability"]}')

    print("\nRecommended Tracks")
    print("-" * 50)

    content_recommendations = content_recommender.recommend(test_song, num_recs=5)
    print(f'\nContent-Based Recommendations:\n')
    print_recommendations(content_recommendations)

    rule_recommendations = rule_recommender.recommend(test_song, num_recs=5)
    print(f'\nRule-Based Recommendations:\n')
    print_recommendations(rule_recommendations)
