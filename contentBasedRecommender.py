from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

class ContentBasedRecommender:
    # Initierar med df
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None        
        self.scaler = StandardScaler()
        self.audio_features = None
        self.track_indices = None

    # Hämtar info om en specifik låt
    def track_info(self, track_name):
        if track_name not in self.track_indices:
            return "Track not found in the dataset."
        
        # Hämtar sångens rad
        index = self.track_indices[track_name]
        track_data = self.df.iloc[index]
        return track_data

    # Skapar tfidf matrix och skalar audio features
    def fit(self):
        # Skapar tfidf matrix för sångnamnen
        vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = vectorizer.fit_transform(self.df['track_name'])

        # Skalar audio features
        audio_features = ['tempo', 'loudness', 'danceability']
        self.audio_features = self.scaler.fit_transform(self.df[audio_features])

        # Skapar en serie för att hitta index baserat på sångnamn
        self.track_indices = pd.Series(self.df.index, index=self.df['track_name'])

    # Rekommenderar låtar baserat på content och audio features
    def recommend(self, track_name, num_recs=5):
        if track_name not in self.track_indices:
            return f"Track '{track_name}' not found in the dataset."

        index = self.track_indices[track_name]

        # Räknar cosine similarity för sång namn och euclidean distances för audio features
        track_vector = self.tfidf_matrix[index]
        similarity_scores = cosine_similarity(track_vector, self.tfidf_matrix).flatten()

        track_audio_vector = self.audio_features[index].reshape(1, -1)
        audio_distances = euclidean_distances(track_audio_vector, self.audio_features).flatten()
        max_distance = np.max(audio_distances)
        audio_similarity_scores = 1 - (audio_distances / max_distance)

        # Kombinerar audio features och namn similarity
        combined_similarity = (similarity_scores + audio_similarity_scores) / 2

        # Sorterar sångarna baserat på kombinerad similarity
        scores = list(enumerate(combined_similarity))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Filtrerar bort test_song från rekommendationerna
        scores = [s for s in scores if s[0] != index]
        top_indices = [s[0] for s in scores[:num_recs]]

        # Hämtar rekommendationer 
        recommendations = []
        for i in top_indices:
            track_data = self.df.iloc[i]
            similarity_score = combined_similarity[i]
            recommendations.append({
                'track_name': track_data['track_name'],
                'track_artist': track_data['track_artist'],
                'track_popularity': track_data['track_popularity'],
                'similarity_score': round(similarity_score, 3),
                'tempo': track_data['tempo'],
                'loudness': track_data['loudness'],
                'danceability': track_data['danceability']
            })
        return recommendations