import numpy as np
import pandas as pd

class EvaluationVerification:
    def __init__(self, df):
        self.df = df

    def mean_feature_distance(self, recommendations, test_track):
        recommendations = pd.DataFrame(recommendations)
        features = ['tempo', 'loudness', 'danceability']
        test_vec = test_track[features].values.astype(float)
        distances = []
        for _, rec in recommendations.iterrows():
            rec_vec = rec[features].values.astype(float)
            distances.append(np.linalg.norm(test_vec - rec_vec))
        return np.mean(distances)

    def mean_absolute_error(self, recommendations, test_track):
        recommendations = pd.DataFrame(recommendations)
        features = ['tempo', 'loudness', 'danceability']
        test_vec = test_track[features].values.astype(float)
        errors = []
        for _, rec in recommendations.iterrows():
            rec_vec = rec[features].values.astype(float)
            errors.append(np.abs(test_vec - rec_vec))
        return np.mean(errors)

    def mean_feature_correlation(self, recommendations, test_track):
        recommendations = pd.DataFrame(recommendations)
        features = ['tempo', 'loudness', 'danceability']
        test_vec = test_track[features].values.astype(float)
        correlations = []
        for _, rec in recommendations.iterrows():
            rec_vec = rec[features].values.astype(float)
            corr = np.corrcoef(test_vec, rec_vec)[0, 1]
            correlations.append(corr)
        return np.mean(correlations)