import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Data Collection and Loading
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    if not os.path.exists(os.path.join(DATA_DIR, "ml-100k")):
        print("Downloading MovieLens 100K dataset...")
        r = requests.get(MOVIELENS_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Download complete.")

download_data()

# Load Data
ratings_path = os.path.join(DATA_DIR, "ml-100k", "u.data")
movies_path = os.path.join(DATA_DIR, "ml-100k", "u.item")

ratings = pd.read_csv(
    ratings_path,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)
movies = pd.read_csv(
    movies_path,
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"],
)

print("Initial Ratings Data:\n", ratings.head())
print("\nInitial Movies Data:\n", movies.head())

# Data Understanding
print("\n" + "="*80)
print("Dataset Info:")
print("="*80)
print("\nRatings Dataset Info:")
print(ratings.info())
print("\nMovies Dataset Info:")
print(movies.info())

# Data Preprocessing
print("\n" + "="*80)
print("Data Preprocessing")
print("="*80)
print("\nProcessing data...")
# Build user-movie matrix
matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
print("\nUser-Movie Matrix Shape:", matrix.shape)
print("\nProcessed Matrix (First 5 users, First 10 movies):")
print(matrix.iloc[:5, :10])

# Similarity Computation
print("\n" + "="*80)
print("Similarity Computation")
print("="*80)
print("\nComputing movie similarities...")
item_sim = cosine_similarity(matrix.T)
item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
print("\nSimilarity Matrix (First 10 movies):")
print(item_sim_df.iloc[:10, :10])

# Statistical Analysis
print("\n" + "="*80)
print("Statistical Summary")
print("="*80)
print("\nStatistical Summary of Ratings:")
print(ratings.describe())
print("\nRating Distribution:")
print(ratings['rating'].value_counts().sort_index())

# Similarity Matrix Analysis
print("\n" + "="*80)
print("Similarity Matrix Analysis")
print("="*80)
print("\nSimilarity Matrix Statistics:")
print(item_sim_df.describe())
movie_id = 1
print(f"\nTop 10 Movies similar to movie {movie_id}:")
similar_movies = item_sim_df[movie_id].sort_values(ascending=False).head(11)
print(similar_movies)

# Correlation Analysis
print("\n" + "="*80)
print("Correlation Matrix")
print("="*80)
# Create a correlation matrix for user ratings
user_rating_stats = ratings.groupby('user_id').agg({
    'rating': ['mean', 'std', 'count']
}).reset_index()
user_rating_stats.columns = ['user_id', 'avg_rating', 'std_rating', 'num_ratings']
print("\nUser Rating Statistics Correlation:")
print(user_rating_stats[['avg_rating', 'std_rating', 'num_ratings']].corr())

# Data Visualization

# 1. Heatmap - Similarity matrix for top movies
plt.figure(figsize=(12, 10))
top_movies = item_sim_df.iloc[:20, :20]
sns.heatmap(top_movies, annot=False, cmap='coolwarm')
plt.title("Movie Similarity Heatmap")
plt.show()

# 2. Histogram - Rating distribution
plt.figure()
plt.hist(ratings['rating'], bins=5, edgecolor='black', color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

# 3. Box Plot - Rating spread
plt.figure()
plt.boxplot(ratings['rating'])
plt.ylabel('Rating')
plt.title('Rating Distribution Box Plot')
plt.show()

# 4. Bar Chart - Average ratings per movie (top 10)
top_rated = ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(10)
plt.figure()
top_rated.plot(kind='bar', color='coral')
plt.xlabel('Movie ID')
plt.ylabel('Average Rating')
plt.title('Top 10 Movies by Average Rating')
plt.show()

# 5. Scatter Plot - Number of ratings vs average rating
movie_stats = ratings.groupby('movie_id').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['avg_rating', 'num_ratings']
plt.figure()
plt.scatter(movie_stats['num_ratings'], movie_stats['avg_rating'], alpha=0.5)
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.title('Number of Ratings vs Average Rating')
plt.show()

# 6. User activity distribution
user_activity = ratings.groupby('user_id').size()
plt.figure()
plt.hist(user_activity, bins=30, edgecolor='black', color='lightgreen')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Frequency')
plt.title('User Activity Distribution')
plt.show()

# 7. Rating trends over time
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings_by_month = ratings.groupby(ratings['timestamp'].dt.to_period('M')).size()
plt.figure(figsize=(12, 6))
ratings_by_month.plot(kind='line', color='purple')
plt.xlabel('Month')
plt.ylabel('Number of Ratings')
plt.title('Rating Trends Over Time')
plt.show()

# 8. KDE Plot - Density plot of ratings
plt.figure()
sns.kdeplot(ratings['rating'], fill=True)
plt.title("Density Plot of Ratings")
plt.xlabel('Rating')
plt.show()

# 9. Correlation between user activity and average rating
user_stats = ratings.groupby('user_id').agg({'rating': ['mean', 'count']})
user_stats.columns = ['avg_rating', 'num_ratings']
plt.figure()
plt.scatter(user_stats['num_ratings'], user_stats['avg_rating'], alpha=0.5, color='orange')
plt.xlabel('Number of Ratings by User')
plt.ylabel('Average Rating Given by User')
plt.title('User Activity vs Average Rating Given')
plt.show()

# 10. Top 10 most rated movies
most_rated = ratings.groupby('movie_id').size().sort_values(ascending=False).head(10)
most_rated_titles = movies[movies['movie_id'].isin(most_rated.index)].set_index('movie_id')
most_rated_titles['count'] = most_rated
plt.figure(figsize=(10, 6))
plt.barh(most_rated_titles['title'], most_rated_titles['count'], color='teal')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Most Rated Movies')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("Implementation Complete!")
print("="*50)
print(f"Total Ratings: {len(ratings)}")
print(f"Total Movies: {len(movies)}")
print(f"Total Users: {ratings['user_id'].nunique()}")
print(f"Rating Range: {ratings['rating'].min()} to {ratings['rating'].max()}")
print(f"Average Rating: {ratings['rating'].mean():.2f}")
print(f"Matrix Sparsity: {(1 - (len(ratings) / (matrix.shape[0] * matrix.shape[1]))) * 100:.2f}%")
