
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# Define the base directory
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # one level up from scripts/
DATA_DIR = BASE_DIR / "data"


# Load the CSVs
tracks = pd.read_csv(DATA_DIR / "tracks.csv")
artists = pd.read_csv(DATA_DIR / "artists.csv")



# made sure on both csvs that each column in dataset only contains one number
#print(tracks['id_artists'].head(10))
#ensure they are both the same type before merging and convert to string
# print(tracks['id_artists'].dtype)
# print(artists['id'].dtype)
tracks['id_artists'] = tracks['id_artists'].astype(str)
artists['id'] = artists['id'].astype(str)

# ids print out differently, tracks csv are surronded in [''] need to clean
#print(tracks['id_artists'].head(20))
#print(artists['id'].head(20))


# making sure the merge worked- when i ran this it showed that five additional columns were added to dataframe
# print(f"Tracks shape: {tracks.shape}")
# print(f"Artists shape: {artists.shape}")
# print(f"Merged shape: {merged_df.shape}")
# decided to do inner join instead of left join for a cleaner dataset
# this way the data that doesn't have a match for id will drop
#merged_df = tracks.merge(artists, left_on='id_artists', right_on='id', how='inner')

#print(merged_df.head(15))

# Remove square brackets and single quotes
tracks['id_artists'] = tracks['id_artists'].str.strip("[]'")

# make sure its a string
tracks['id_artists'] = tracks['id_artists'].astype(str)

#double check that brackets are gone
# print(tracks['id_artists'].head(20))
# print(artists['id'].head(20))
#
#
merged_df = tracks.merge(artists, left_on='id_artists', right_on='id', how='inner')
# print(merged_df.head(15))
# print(f"Merged shape: {merged_df.shape}")


# print(merged_df.columns)
# need to handle multiple artists in one song, maybe convert to list
# print(merged_df['artists'].head(10))

# merged_df.to_csv('merged_df.csv', index=False)  this code can make it into a csv
#
# Look for text-based nulls or blanks
# print(merged_df.info())

# Ultimately the dataset is now cleaned with 0 missing values and 0 duplicate values
# inner join dropped the columns where the artist id had no matching artist or track info
# So after the join, the merged dataframe no longer has rows with NaN in any of those key columns

# now i need to rename columns for clarity - delete duplicate columns if they exist


# Rename columns for clarity
df2 = merged_df.rename(columns={
    # Track info
    "id_x": "track_id",
    "name_x": "track_name",
    "popularity_x": "track_popularity",
    "duration_ms": "track_duration_ms",
    "explicit": "explicit_lyrics",
    "artists": "track_artists",
    "id_artists": "artist_id",
    "release_date": "release_date",
    "danceability": "danceability",
    "energy": "energy",
    "key": "musical_key",
    "loudness": "loudness",
    "mode": "mode",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "valence": "valence",
    "tempo": "tempo",
    "time_signature": "time_signature",

    # Artist info
    "id_y": "artist_id_dup",  # (optional: drop later since same as artist_id)
    "followers": "artist_followers",
    "genres": "artist_genres",
    "name_y": "artist_name",
    "popularity_y": "artist_popularity"
})

df2['release_date'] = pd.to_datetime(df2['release_date'], errors='coerce')

# Extract release year
df2['release_year'] = df2['release_date'].dt.year

# Group by year and calculate mean followers (to smooth it out)
followers_by_year = df2.groupby('release_year')['artist_followers'].mean().reset_index()

# Plot
plt.figure(figsize=(10,6))
plt.plot(followers_by_year['release_year'], followers_by_year['artist_followers'], marker='o')

plt.title("Average Artist Followers vs Release Year")
plt.xlabel("Release Year")
plt.ylabel("Average Followers")
plt.grid(True)
plt.show()

followers_by_year = df2.groupby('release_year')['artist_followers'].agg(['mean', 'median']).reset_index()

plt.figure(figsize=(12,6))
plt.plot(followers_by_year['release_year'], followers_by_year['mean'], label='Mean Followers', marker='o')
plt.plot(followers_by_year['release_year'], followers_by_year['median'], label='Median Followers', marker='o')

plt.title("Artist Followers vs Release Year (Mean vs Median)")
plt.xlabel("Release Year")
plt.ylabel("Followers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# datetime thorugh pandas
#df2['release_date'] = pd.to_datetime(df2['release_date'], errors='coerce')

# extract just the year
#df2['release_year'] = df2['release_date'].dt.year

# filtering for songs released after 2012
#df_post2012 = df2[df2['release_year'] > 2012]

# print(df_post2012.describe())

# irrelevant_cols = [
 ##   "track_artists", "artist_id", "release_date", "artist_id_dup",
 #   "artist_followers", "artist_genres", "artist_name", "artist_popularity"
# ]

# clean_df = df_post2012.drop(columns=irrelevant_cols)

# print(clean_df.columns)

# Export to CSV (without the index column)
# clean_df.to_csv("spotify_post2012.csv", index=False)
