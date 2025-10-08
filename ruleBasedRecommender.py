import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class RuleBasedRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.track_indices = pd.Series(self.df.index, index=self.df['track_name'])
        self.features = ['tempo', 'loudness', 'danceability', 'track_popularity']


    def calculate_combined_score(self, seed_track_scaled):
        score_tempo = 1 - np.abs(self.df['tempo'] - seed_track_scaled['tempo'])
        score_loudness = 1 - np.abs(self.df['loudness'] - seed_track_scaled['loudness'])
        score_danceability = 1 - np.abs(self.df['danceability'] - seed_track_scaled['danceability'])
        score_popularity = self.df['track_popularity'] * 0.5 

        similarity_score = (score_tempo + score_loudness + score_danceability + score_popularity)
        
        return similarity_score

    def recommend(self, track_name, num_recs=5):
        if track_name not in self.track_indices:
            return f"Track '{track_name}' not found."

        seed_track_original = self.df[self.df['track_name'] == track_name].iloc[0]
        seed_track_scaled = seed_track_original[self.features] 
        
        self.df['similarity_score'] = self.calculate_combined_score(seed_track_scaled)
        self.df['similarity_score'] = MinMaxScaler().fit_transform(self.df[['similarity_score']])

        recommendations_df = self.df.sort_values(by='similarity_score', ascending=False)
        recommendations_df = recommendations_df[recommendations_df['track_name'] != track_name]

        recommendations = []
        top_recommendations = recommendations_df.head(num_recs)
        
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                'track_name': row['track_name'],
                'track_artist': row['track_artist'],
                'track_popularity': row['track_popularity'],
                'similarity_score': round(float(row['similarity_score']), 3),
                'loudness': row['loudness'],
                'tempo': row['tempo'],
                'danceability': row['danceability']
            })
        
        return recommendations
