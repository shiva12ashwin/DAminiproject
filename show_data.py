import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load ratings data
ratings_path = os.path.join(DATA_DIR, "ml-100k", "u.data")
ratings = pd.read_csv(
    ratings_path,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

# Load movies data
movies_path = os.path.join(DATA_DIR, "ml-100k", "u.item")
movies = pd.read_csv(
    movies_path,
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"],
)

print("=" * 80)
print("MOVIELENS 100K DATASET - RATINGS DATA (u.data)")
print("=" * 80)
print("\nFirst 15 rows of ratings dataset:")
print(ratings.head(15).to_string(index=False))

print("\n\n" + "=" * 80)
print("MOVIELENS 100K DATASET - MOVIES DATA (u.item)")
print("=" * 80)
print("\nFirst 15 rows of movies dataset:")
print(movies.head(15).to_string(index=False))

print("\n\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)
print(f"Total Ratings: {len(ratings)}")
print(f"Total Movies: {len(movies)}")
print(f"Total Users: {ratings['user_id'].nunique()}")
print(f"Rating Range: {ratings['rating'].min()} to {ratings['rating'].max()}")
print(f"Average Rating: {ratings['rating'].mean():.2f}")
print("=" * 80)
