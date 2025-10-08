import pandas as pd

class HybridRecommender:
    def __init__(self, content_recs=None, rule_recs=None):
        self.content_recs = content_recs
        self.rule_recs = rule_recs  

    def recommend(self, track_name, num_recs=5):
        content_recs = self.content_recs
        rule_recs = self.rule_recs

        combined_recs = {}
        max_length = max(len(content_recs), len(rule_recs))
        for i in range(max_length):
            if i < len(content_recs):
                rec = content_recs[i]
                if rec['track_name'] not in combined_recs:
                    combined_recs[rec['track_name']] = rec

            if i < len(rule_recs):
                rec = rule_recs[i]
                if rec['track_name'] not in combined_recs:
                    combined_recs[rec['track_name']] = rec

            if len(combined_recs) >= num_recs:
                break
        
        hybrid_recs = list(combined_recs.values())
        hybrid_recs = hybrid_recs[:num_recs]
        #combined_recs = {rec['track_name']: rec for rec in content_recs + rule_recs}
        #ranked_recs = sorted(combined_recs.values(), key=lambda x: (x['similarity_score'], x['track_popularity']), reverse=True)

        return hybrid_recs
