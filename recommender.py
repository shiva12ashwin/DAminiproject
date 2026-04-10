import os
import zipfile
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


def load_data():
    download_data()
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
    return ratings, movies


def build_model():
    ratings, movies = load_data()

    # Build user-movie matrix
    matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

    # Item-based cosine similarity
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

    return ratings, movies, matrix, item_sim_df


# Load once at import time
ratings, movies, user_movie_matrix, item_similarity = build_model()


def get_all_movies():
    """Return list of all movies for the dropdown."""
    return movies[["movie_id", "title"]].sort_values("title").to_dict(orient="records")


def recommend_by_movie(movie_id: int, top_n: int = 10):
    """Item-based CF: find movies similar to the given movie."""
    movie_id = int(movie_id)
    if movie_id not in item_similarity.columns:
        return []

    sim_scores = item_similarity[movie_id].drop(index=movie_id).sort_values(ascending=False)
    top_ids = sim_scores.head(top_n).index.tolist()

    result = movies[movies["movie_id"].isin(top_ids)].copy()
    result["score"] = result["movie_id"].map(sim_scores)
    result = result.sort_values("score", ascending=False)
    return result[["movie_id", "title", "score"]].to_dict(orient="records")


def recommend_for_user(user_id: int, top_n: int = 10):
    """User-based CF: recommend movies the user hasn't seen yet."""
    user_id = int(user_id)
    if user_id not in user_movie_matrix.index:
        return []

    user_ratings = user_movie_matrix.loc[user_id]
    seen = user_ratings[user_ratings > 0].index.tolist()

    # Weighted sum of item similarities for unseen movies
    scores = item_similarity[seen].sum(axis=1)
    scores = scores.drop(index=seen, errors="ignore")
    scores = scores.sort_values(ascending=False).head(top_n)

    result = movies[movies["movie_id"].isin(scores.index)].copy()
    result["score"] = result["movie_id"].map(scores)
    result = result.sort_values("score", ascending=False)
    return result[["movie_id", "title", "score"]].to_dict(orient="records")


def get_user_ids():
    return sorted(ratings["user_id"].unique().tolist())
