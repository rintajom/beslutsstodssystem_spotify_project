import pandas as pd
from contentBasedRecommender import ContentBasedRecommender
from hybridRecommender import HybridRecommender
from ruleBasedRecommender import RuleBasedRecommender
from evaluationVerification import EvaluationVerification

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_spotify_data.csv")
    content_recommender = ContentBasedRecommender(df)
    rule_recommender = RuleBasedRecommender(df)

    test_song = "Good Luck, Babe!"

    # Printar rekommendationer i lättare läst formatering
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

    # info om test_song
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

    # Hämtar conten-based rekommendationer
    content_recommendations = content_recommender.recommend(test_song, num_recs=5)
    print(f'\nContent-Based Recommendations:\n')
    print_recommendations(content_recommendations)

    # Hämtar rule-based rekommendationer
    rule_recommendations = rule_recommender.recommend(test_song, num_recs=5)
    print(f'\nRule-Based Recommendations:\n')
    print_recommendations(rule_recommendations)

    # Hämtar hybrid rekommendationer
    hybrid_recommender = HybridRecommender(content_recommendations, rule_recommendations)
    hybrid_recommendations = hybrid_recommender.recommend(test_song, num_recs=5)
    print(f'\nHybrid Recommendations:\n')
    print_recommendations(hybrid_recommendations)

    # Hämtar all info för test_song
    test_track = df[df['track_name'] == test_song].iloc[0]

    # Hämtar evaluation värden
    evaluator = EvaluationVerification(df)
    content_feature_dist = evaluator.mean_feature_distance(content_recommendations, test_track)
    rule_feature_dist = evaluator.mean_feature_distance(rule_recommendations, test_track)
    hybrid_feature_dist = evaluator.mean_feature_distance(hybrid_recommendations, test_track)

    content_mae = evaluator.mean_absolute_error(content_recommendations, test_track)
    rule_mae = evaluator.mean_absolute_error(rule_recommendations, test_track)
    hybrid_mae = evaluator.mean_absolute_error(hybrid_recommendations, test_track)

    content_corr = evaluator.mean_feature_correlation(content_recommendations, test_track)
    rule_corr = evaluator.mean_feature_correlation(rule_recommendations, test_track)
    hybrid_corr = evaluator.mean_feature_correlation(hybrid_recommendations, test_track)

    # Printar ut evaluation värden
    print("\nEvaluation Metrics:")
    print("-" * 65)
    print(f"{'Metric':<28} | {'Content-Based':^13} | {'Rule-Based':^11} | {'Hybrid':^9}")
    print("-" * 65)
    print(f"{'Mean Feature Distance':<28} | {content_feature_dist:^13.4f} | {rule_feature_dist:^11.4f} | {hybrid_feature_dist:^9.4f}")
    print(f"{'Mean Absolute Error':<28} | {content_mae:^13.4f} | {rule_mae:^11.4f} | {hybrid_mae:^9.4f}")
    print(f"{'Mean Feature Correlation':<28} | {content_corr:^13.4f} | {rule_corr:^11.4f} | {hybrid_corr:^9.4f}")
    print("-" * 65) 