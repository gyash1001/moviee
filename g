import pandas as pd
import numpy as np

# warnings
import warnings
warnings.filterwarnings('ignore')


# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
df_netflix = pd.read_csv('netflixtitles.csv')
df_amazon =  pd.read_csv('amazontitles.csv')
df_hbo =  pd.read_csv('hbotitles.csv')
df = pd.concat([df_netflix, df_amazon, df_hbo], axis=0)
df_movies = df.drop_duplicates()
# Drop unnecessary columns
df_movies.drop(['description', 'age_certification'], axis=1, inplace=True)
df['production_countries']
df_movies['production_countries'] = df_movies['production_countries'].str.replace(r"\[", '', regex=True).str.replace(r"'", '', regex=True).str.replace(r"\]", '', regex=True)
df_movies['lead_prod_country'] = df_movies['production_countries'].str.split(',').str[0]
df_movies['prod_countries_cnt'] = df_movies['production_countries'].str.split(',').str.len()
df_movies['lead_prod_country'] = df_movies['lead_prod_country'].replace('', np.nan)
df_movies['lead_prod_country']
df_movies['genres']
df_movies['genres'] = df_movies['genres'].str.replace(r"\[", '', regex=True).str.replace(r"'", '', regex=True).str.replace(r"\]", '', regex=True)
df_movies['main_genre'] = df_movies['genres'].str.split(',').str[0]
df_movies['main_genre'] = df_movies['main_genre'].replace('', np.nan)
df_movies['main_genre']

df_movies.drop(['genres', 'production_countries'], axis=1, inplace=True)
df_movies.shape
df_movies.isnull().sum()
# Drop rows with any missing values to clean the dataset
df_movies.dropna(inplace=True)

# Set the 'title' column as the DataFrame index
df_movies.set_index('title', inplace=True)

# Drop the 'id' and 'imdb_id' columns as they are not needed for further analysis
df_movies.drop(['id', 'imdb_id'], axis=1, inplace=True)
df_movies.shape
# Create dummy variables for categorical columns ('type', 'lead_prod_country', 'main_genre')
dummies = pd.get_dummies(df_movies[['type', 'lead_prod_country', 'main_genre']], drop_first=True)

# Concatenate the dummy variables with the original DataFrame
df_movies_dum = pd.concat([df_movies, dummies], axis=1)

# 14. Drop the original categorical columns after creating dummy variables
df_movies_dum.drop(['type', 'lead_prod_country', 'main_genre'], axis=1, inplace=True)
# Apply MinMaxScaler to scale the data for model training
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_movies_dum)
df_scaled = pd.DataFrame(df_scaled, columns=df_movies_dum.columns)

# Display the scaled DataFrame

df_scaled
# Define the range of epsilon (eps) and minimum samples (min_samples) parameters for DBSCAN
eps_array = [0.2, 0.5, 1]  # List of different epsilon values (the maximum distance between two samples for one to be considered as in the neighborhood of the other)
min_samples_array = [5, 10, 30]  # List of different min_samples values (the number of samples in a neighborhood for a point to be considered as a core point)

# Iterate over each combination of eps and min_samples
for eps in eps_array:
    for min_samples in min_samples_array:
        # Initialize and fit the DBSCAN model with the current parameters
        clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(df_scaled)

        # Retrieve the cluster labels from the fitted model
        cluster_labels = clusterer.labels_

        # Check if the algorithm found only one cluster or marked all points as noise (-1 label for noise)
        if len(set(cluster_labels)) == 1:
            continue  # Skip this combination as it does not provide meaningful clusters

        # Calculate the silhouette score to evaluate the quality of the clustering
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)

        # Print the current parameters, number of clusters, and the silhouette score
        print("For eps =", eps,
              "For min_samples =", min_samples,
              "Count clusters =", len(set(cluster_labels)),
              "The average silhouette_score is :", silhouette_avg)

dbscan_model = DBSCAN(eps=1, min_samples=5).fit(df_scaled)
print("For eps =", 1,
      "For min_samples =", 5,
      "Count clusters =", len(set(dbscan_model.labels_)),
      "The average silhouette_score is :", silhouette_score(df_scaled, dbscan_model.labels_))
df_movies['dbscan_clusters'] = dbscan_model.labels_
df_movies['dbscan_clusters'].value_counts()

import random

def recommend_movie(movie_name: str):
    # Convert the input movie name to lowercase for case-insensitive matching
    movie_name = movie_name.lower()

    # Create a new column 'name' with lowercase movie names for comparison
    df_movies['name'] = df_movies.index.str.lower()

    # Find the movie that matches the input name
    movie = df_movies[df_movies['name'].str.contains(movie_name, na=False)]

    if not movie.empty:
        # Get the cluster label of the input movie
        cluster = movie['dbscan_clusters'].values[0]

        # Get all movies in the same cluster
        cluster_movies = df_movies[df_movies['dbscan_clusters'] == cluster]

        # If there are more than 5 movies in the cluster, randomly select 5
        if len(cluster_movies) >= 5:
            recommended_movies = random.sample(list(cluster_movies.index), 5)
        else:
            # If fewer than 5, return all the movies in the cluster
            recommended_movies = list(cluster_movies.index)

        # Print the recommended movies
        print('--- We can recommend you these movies ---')
        for m in recommended_movies:
            print(m)
    else:
        print('Movie not found in the database.')
s = input('Input movie name: ')

print("\n\n")
recommend_movie(s)
