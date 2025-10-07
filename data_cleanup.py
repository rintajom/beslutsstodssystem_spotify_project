import pandas as pd

data = pd.read_csv("data/spotify_data.csv") # hämtad från kagglehub

# droppar onödiga kolumner
df = data.drop(columns=["valence", "time_signature", "track_href", "uri", "analysis_url", "liveness", "speechiness", "mode", "key", "duration_ms", "acousticness", "playlist_subgenre", "playlist_id", "playlist_genre", "playlist_name", "track_album_id", "track_id", "instrumentalness", "id", "energy", "type"])

# egen id kolumn
df['id'] = df.index

# ordnar kolumner logiskt
df = df[['id', 'track_name', 'track_artist', 'track_album_name', 'track_album_release_date', 'track_popularity', 'tempo', 'loudness', 'danceability']]

# duplicates 
df = df.drop_duplicates(subset=['track_name'])

# tomma värden
df = df.dropna()
df = df.reset_index(drop=True)

df.info()
print(df.head())

# sparar nya dataframen som csv
df.to_csv("data/cleaned_spotify_data.csv", index=False)