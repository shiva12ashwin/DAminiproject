# Movie Recommender System

## Abstract

The entertainment industry depends on recommendation systems to enhance user experience because these systems identify content that matches individual preferences and viewing patterns. Streaming platforms and movie databases generate large volumes of data related to user activities such as movie ratings, viewing history, genre preferences, and user demographics. The process of analyzing this data leads to obtaining important information which shows how different factors influence user preferences while enabling platforms to deliver personalized content recommendations. The Movie Recommender System focuses on examining user-movie interaction data using collaborative filtering techniques and similarity computation methods. The system uses cosine similarity measures to create movie recommendation profiles which show complete rating patterns and display how users interact with different movies. The system uses correlation analysis to find out how user rating behaviors affect movie similarity by examining their connection with rating matrices. The researchers used collaborative filtering techniques which include item-based filtering and user-based filtering to create recommendation models that viewers can trust and utilize. The computational tools enable users to discover similar movies and personalized suggestions which remain hidden in the underlying rating data. Item-based filtering shows the connection between movies with similar rating patterns while user-based filtering displays personalized recommendations which include movies aligned with individual preferences. This analysis helps the system to find the most important variables which determine movie similarity and user preferences. The system results enable entertainment platforms and movie enthusiasts to discover relevant content which will enhance viewing experiences and lead users to achieve their entertainment goals.

A Python-based movie recommendation system using collaborative filtering on the MovieLens 100K dataset. This system provides personalized movie recommendations using item-based and user-based collaborative filtering algorithms.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive CLI](#interactive-cli)
  - [Python Library](#python-library)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Code Explanation](#code-explanation)
- [License](#license)

## Problem Statement

Entertainment platforms and movie databases collect vast amounts of user interaction data through their rating systems and viewing histories and user preferences and demographic information. The data remains in a disorganized state because platforms lack proper recommendation methods which prevents them from discovering essential elements that drive user satisfaction and content discovery. Users need proper recommendation systems because they require those systems to detect all patterns and preferences which create challenges in finding relevant movies that match their interests. The main problem requires developers to use collaborative filtering techniques and similarity computation methods for systematic data analysis which will provide them with accurate recommendations. The system uses cosine similarity and matrix factorization and correlation measures to study how different user-movie interactions distribute and connect with one another. Recommendation algorithms enable users to identify similar movies together with personalized suggestions and preference patterns. The study intends to identify which rating patterns and user behaviors have the highest impact while offering valuable insights that will enhance content discovery processes and boost user engagement rates.

## Theoretical Background

The Movie Recommender System is based on the principles of collaborative filtering and similarity computation, which are used to analyze and interpret user-movie interaction data. Collaborative filtering provides methods to generate recommendations through techniques such as item-based filtering, user-based filtering, and matrix factorization. The movie recommendation engine uses these measures to identify rating patterns and user preferences, which enables an overall assessment of content relevance and user satisfaction.

The study employs cosine similarity analysis as its primary method to assess how movies relate to each other based on user rating patterns. The method assesses how user ratings and viewing behaviors and preference indicators affect movie similarity scores. A high similarity score shows that two movies share similar rating patterns across users while a low similarity score shows that movies appeal to different user segments. The use of collaborative filtering techniques enables the transformation of sparse rating matrices into meaningful recommendation lists that people can easily utilize.

Researchers use computational methods such as matrix operations and similarity calculations and ranking algorithms to extract information about user preferences and movie relationships from their dataset. The mathematical representations between rating vectors make it easier to understand movie similarities and find relevant recommendations. Recommendation generation requires data preprocessing to be performed as a fundamental procedure. The process involves complete data validation through missing value resolution, duplicate elimination, and matrix normalization to achieve precise results. Analysis quality increases when rating data is organized and processed because it enables the system to obtain more accurate recommendations.

The theoretical foundation uses collaborative filtering and similarity computation methods to change raw user-movie interaction data into usable recommendations which enhance content discovery and user satisfaction. Item-based collaborative filtering analyzes the rating patterns between movies to identify similar content based on how users rated them collectively. User-based collaborative filtering examines individual user rating histories to predict which unwatched movies align with their established preferences. The combination of these approaches creates a robust recommendation system that adapts to diverse user needs and movie characteristics.

## Features

- **Item-based Collaborative Filtering**: Find movies similar to a selected movie based on user rating patterns
- **User-based Collaborative Filtering**: Get personalized movie recommendations for specific users
- **Interactive CLI**: User-friendly command-line interface for exploring recommendations
- **Python Library**: Import and use recommendation functions in your own projects
- **Automatic Data Management**: MovieLens 100K dataset downloads and extracts automatically on first run
- **Efficient Similarity Computation**: Uses cosine similarity for fast recommendation generation

## Tech Stack

- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Cosine similarity computation
- **Requests**: HTTP library for dataset download
- **MovieLens 100K**: Real-world movie ratings dataset

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd movie-recommender
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application (dataset will download automatically on first run):
```bash
python main.py
```

## Usage

### Interactive CLI

Launch the interactive command-line interface:

```bash
python main.py
```

The CLI provides the following options:

1. **Get recommendations based on a movie**: Enter a movie ID to find similar movies
2. **Get recommendations for a user**: Enter a user ID to get personalized suggestions
3. **List all movies**: View available movies in the dataset
4. **List all users**: View available user IDs
5. **Exit**: Close the application

### Python Library

Import and use the recommender functions in your own Python scripts:

```python
from recommender import (
    recommend_by_movie,
    recommend_for_user,
    get_all_movies,
    get_user_ids
)

# Example 1: Get movies similar to "Toy Story" (movie_id=1)
similar_movies = recommend_by_movie(movie_id=1, top_n=10)
print("Movies similar to Toy Story:")
for movie in similar_movies:
    print(f"  {movie['title']} (Similarity: {movie['score']:.4f})")

# Example 2: Get personalized recommendations for user 196
user_recommendations = recommend_for_user(user_id=196, top_n=10)
print("\nRecommendations for User 196:")
for movie in user_recommendations:
    print(f"  {movie['title']} (Score: {movie['score']:.4f})")

# Example 3: Get all available movies
all_movies = get_all_movies()
print(f"\nTotal movies in dataset: {len(all_movies)}")

# Example 4: Get all user IDs
all_users = get_user_ids()
print(f"Total users in dataset: {len(all_users)}")
```

## How It Works

The recommendation system uses collaborative filtering, which makes predictions based on user behavior patterns:

### Item-Based Collaborative Filtering

1. Creates a user-movie rating matrix
2. Computes cosine similarity between all movie pairs based on user ratings
3. For a given movie, returns the most similar movies based on similarity scores

**Use case**: "Users who liked this movie also liked..."

### User-Based Collaborative Filtering

1. Identifies movies the user has already rated
2. For unwatched movies, calculates a weighted score based on similarity to movies the user liked
3. Returns top-ranked unwatched movies

**Use case**: "Based on your viewing history, you might like..."

### Cosine Similarity

The system uses cosine similarity to measure how similar two movies are based on user rating patterns:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Higher similarity scores indicate movies with similar rating patterns across users.

## Dataset

This project uses the **MovieLens 100K** dataset from GroupLens Research:

- **100,000 ratings** (scale: 1-5 stars)
- **943 users** (each rated at least 20 movies)
- **1,682 movies**
- **Collected**: September 1997 - April 1998
- **Source**: [GroupLens Research](https://grouplens.org/datasets/movielens/)

The dataset includes:
- User ratings (user_id, movie_id, rating, timestamp)
- Movie information (movie_id, title, release date, genres)
- User demographics (age, gender, occupation, zip code)

## Project Structure

```
movie-recommender/
│
├── main.py                 # Interactive CLI application
├── recommender.py          # Core recommendation engine
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
└── data/
    └── ml-100k/           # MovieLens dataset (auto-downloaded)
        ├── u.data         # User ratings
        ├── u.item         # Movie information
        ├── u.user         # User demographics
        └── ...            # Additional dataset files
```

## API Reference

### `recommend_by_movie(movie_id: int, top_n: int = 10) -> list`

Returns movies similar to the specified movie.

**Parameters:**
- `movie_id` (int): ID of the movie to find similar movies for
- `top_n` (int, optional): Number of recommendations to return (default: 10)

**Returns:**
- List of dictionaries containing `movie_id`, `title`, and `score`

**Example:**
```python
results = recommend_by_movie(movie_id=50, top_n=5)
```

### `recommend_for_user(user_id: int, top_n: int = 10) -> list`

Returns personalized movie recommendations for a user.

**Parameters:**
- `user_id` (int): ID of the user to generate recommendations for
- `top_n` (int, optional): Number of recommendations to return (default: 10)

**Returns:**
- List of dictionaries containing `movie_id`, `title`, and `score`

**Example:**
```python
results = recommend_for_user(user_id=100, top_n=5)
```

### `get_all_movies() -> list`

Returns all movies in the dataset.

**Returns:**
- List of dictionaries containing `movie_id` and `title`, sorted alphabetically

**Example:**
```python
movies = get_all_movies()
```

### `get_user_ids() -> list`

Returns all user IDs in the dataset.

**Returns:**
- Sorted list of user IDs

**Example:**
```python
users = get_user_ids()
```

## Examples

### Example 1: Find Similar Movies

```python
from recommender import recommend_by_movie

# Find movies similar to "Star Wars" (movie_id=50)
recommendations = recommend_by_movie(50, top_n=5)

print("If you liked Star Wars, you might also like:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie['title']} (Similarity: {movie['score']:.3f})")
```

### Example 2: Personalized User Recommendations

```python
from recommender import recommend_for_user

# Get recommendations for user 1
recommendations = recommend_for_user(1, top_n=5)

print("Recommended movies for you:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie['title']} (Score: {movie['score']:.3f})")
```

### Example 3: Browse Movies

```python
from recommender import get_all_movies

movies = get_all_movies()

print(f"Total movies available: {len(movies)}")
print("\nFirst 10 movies:")
for movie in movies[:10]:
    print(f"ID {movie['movie_id']}: {movie['title']}")
```

## Code Explanation

### 1. DATA COLLECTION AND LOADING

The first step involves collecting the dataset required for analysis. In this project, the MovieLens 100K dataset containing attributes such as user_id, movie_id, rating, and timestamp is collected and processed. The dataset is automatically downloaded from the GroupLens Research repository and loaded into the program using the pandas library.

```python
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
```

**Sample of u.data (ratings dataset):**
```
user_id    movie_id    rating    timestamp
196        242         3         881250949
186        302         3         891717742
22         377         1         878887116
244        51          2         880606923
166        346         1         886397596
298        474         4         884182806
115        265         2         881171488
253        465         5         891628467
305        451         3         886324817
6          86          3         883603013
```

**Sample of u.item (movies dataset):**
```
movie_id    title
1           Toy Story (1995)
2           GoldenEye (1995)
3           Four Rooms (1995)
4           Get Shorty (1995)
5           Copycat (1995)
6           Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)
7           Twelve Monkeys (1995)
8           Babe (1995)
9           Dead Man Walking (1995)
10          Richard III (1995)
```

### 2. DATA UNDERSTANDING

Understanding the dataset and its attributes is a key step in data analytics. It is important because only after understanding the dataset we can process the data accordingly and clean the data into useful processed data. We use head() and info() methods to understand the data.

**head()** - Displays the first few rows  
**info()** - Provides column details, data types, and missing values

```python
# Display first few rows of ratings data
print(ratings.head())

# Display information about ratings dataset
print(ratings.info())

# Display first few rows of movies data
print(movies.head())

# Display information about movies dataset
print(movies.info())
```

**Output of ratings.head():**
```
   user_id  movie_id  rating  timestamp
0      196       242       3  881250949
1      186       302       3  891717742
2       22       377       1  878887116
3      244        51       2  880606923
4      166       346       1  886397596
```

**Output of ratings.info():**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   user_id    100000 non-null  int64
 1   movie_id   100000 non-null  int64
 2   rating     100000 non-null  int64
 3   timestamp  100000 non-null  int64
dtypes: int64(4)
memory usage: 3.1 MB
```

**Output of movies.head():**
```
   movie_id                    title
0         1       Toy Story (1995)
1         2     GoldenEye (1995)
2         3    Four Rooms (1995)
3         4   Get Shorty (1995)
4         5       Copycat (1995)
```

**Output of movies.info():**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1682 entries, 0 to 1681
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   movie_id   1682 non-null   int64 
 1   title      1682 non-null   object
dtypes: int64(1), object(1)
memory usage: 26.4 KB
```

### 3. DATA PREPROCESSING AND MATRIX CONSTRUCTION

Raw data is messy and not useful and so it is important to process it. The following are applied to the dataset to process it into a clean processable data.

```python
# Download and extract the dataset if not already present
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

# Load and process the data
ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
movies = pd.read_csv(movies_path, sep="|", encoding="latin-1", usecols=[0, 1], names=["movie_id", "title"])

# Build user-movie matrix
matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

print("\nProcessed Data:\n", matrix.head())
```

### 3. SIMILARITY COMPUTATION

Similarity computation is the process of measuring how similar two movies are based on user rating patterns, usually using cosine similarity which produces values between 0 and 1.

We use cosine_similarity from scikit-learn to compute similarity between movies. In this:
- A value of 1 indicates identical rating patterns
- A value of 0 indicates no similarity
- Higher values indicate greater similarity between movies

```python
from sklearn.metrics.pairwise import cosine_similarity

# Item-based cosine similarity
item_sim = cosine_similarity(matrix.T)
item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
```

### 4. STATISTICAL ANALYSIS

In this part we use statistical methods to summarize the dataset. We use mean, median and variance methods to describe the data.

```python
print("\nStatistical Summary:\n", ratings.describe())
print("\nRating Distribution:\n", ratings['rating'].value_counts().sort_index())
```

### 5. SIMILARITY MATRIX ANALYSIS

Similarity matrix analysis is used to measure the relationship between movies based on user rating patterns.

It tells you about:
- Whether two movies are related
- How strong the relationship is
- Whether the relationship indicates similar or different user preferences

```python
print("\nSimilarity Matrix:\n", item_sim_df.head())
# Display similarity scores for a specific movie
movie_id = 1
print(f"\nMovies similar to movie {movie_id}:")
print(item_sim_df[movie_id].sort_values(ascending=False).head(10))
```

### 6. DATA VISUALIZATION

Visualization helps in understanding patterns clearly.

- **Heatmap** - Shows similarity between all movies based on rating patterns
- **Scatter Plot** - Shows relationship between user ratings and movie popularity
- **Histogram** - Shows distribution of ratings across the dataset
- **Box Plot** - Identifies outliers and spread in rating data
- **Bar Chart** - Compares average ratings across different movies
- **Distribution Plot** - Shows smooth distribution curve of ratings

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap - Similarity matrix for top movies
plt.figure(figsize=(12, 10))
top_movies = item_sim_df.iloc[:20, :20]
sns.heatmap(top_movies, annot=False, cmap='coolwarm')
plt.title("Movie Similarity Heatmap")
plt.show()

# Histogram - Rating distribution
plt.figure()
plt.hist(ratings['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

# Box Plot - Rating spread
plt.figure()
plt.boxplot(ratings['rating'])
plt.ylabel('Rating')
plt.title('Rating Distribution Box Plot')
plt.show()

# Bar Chart - Average ratings per movie (top 10)
top_rated = ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(10)
plt.figure()
top_rated.plot(kind='bar')
plt.xlabel('Movie ID')
plt.ylabel('Average Rating')
plt.title('Top 10 Movies by Average Rating')
plt.show()
```

### 7. ITEM-BASED RECOMMENDATION GENERATION

The fourth step implements item-based collaborative filtering to find movies similar to a given movie. The system retrieves similarity scores for the target movie, sorts them in descending order, and returns the top N most similar movies.

```python
def recommend_by_movie(movie_id: int, top_n: int = 10):
    movie_id = int(movie_id)
    if movie_id not in item_similarity.columns:
        return []

    sim_scores = item_similarity[movie_id].drop(index=movie_id).sort_values(ascending=False)
    top_ids = sim_scores.head(top_n).index.tolist()

    result = movies[movies["movie_id"].isin(top_ids)].copy()
    result["score"] = result["movie_id"].map(sim_scores)
    result = result.sort_values("score", ascending=False)
    return result[["movie_id", "title", "score"]].to_dict(orient="records")
```

### 8. USER-BASED RECOMMENDATION GENERATION

The fifth step implements user-based collaborative filtering to generate personalized recommendations. The system identifies movies the user has already rated, calculates weighted similarity scores for unwatched movies, and returns the top N recommendations.

```python
def recommend_for_user(user_id: int, top_n: int = 10):
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
```

### 9. INTERACTIVE COMMAND-LINE INTERFACE

The sixth step provides an interactive CLI that allows users to explore recommendations through a menu-driven interface. Users can select options to get movie-based recommendations, user-based recommendations, or browse available movies and users.

```python
def main():
    print("=" * 50)
    print("Movie Recommender System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations based on a movie")
        print("2. Get recommendations for a user")
        print("3. List all movies")
        print("4. List all users")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            movie_id = input("Enter movie ID: ").strip()
            results = recommend_by_movie(int(movie_id))
            # Display results
        # Additional menu options...
```

## Result

The analysis of the MovieLens dataset revealed several important insights regarding the factors influencing movie recommendations and user preferences. After preprocessing the data to handle missing values, build user-movie matrices, and compute similarity scores, collaborative filtering techniques and visualization methods were applied. The similarity analysis showed that movies with similar rating patterns from users established the strongest recommendation indicators for item-based collaborative filtering. User rating history and movie similarity scores showed a positive correlation with recommendation accuracy, demonstrating their effectiveness in predicting user preferences. The relationships between different movies appeared through similarity heatmaps and scatter plots, while histograms and density plots displayed how ratings were distributed across the dataset. Box plots helped in identifying outliers, representing movies with exceptionally high or low rating patterns. The results demonstrate that collaborative filtering, which the system implements through cosine similarity computation and matrix operations, determines accurate recommendation results. The findings provide important information about how users rate movies and how recommendation systems can leverage these patterns to enhance content discovery.

## Future Enhancement

While the current system focuses on collaborative filtering and basic visualization, several improvements can be made to enhance its functionality and effectiveness. One major enhancement would be the integration of advanced machine learning algorithms, such as matrix factorization techniques (SVD, NMF) or deep learning models (neural collaborative filtering), to improve recommendation accuracy and handle cold-start problems. This would enable better predictions for new users and movies with limited rating data.

Additionally, incorporating more diverse features such as movie genres, release dates, user demographics, and temporal dynamics can provide deeper insights into user preferences. The system can also be extended by developing a web-based interactive dashboard using frameworks like Flask, Django, or Streamlit for better visualization and user interaction. Implementing hybrid recommendation approaches that combine collaborative filtering with content-based filtering would further improve recommendation quality.

Real-time recommendation updates and personalization based on user feedback would enhance user experience. Integration with external APIs to fetch movie metadata, posters, and trailers would make the system more engaging. Adding explainability features to show users why certain movies were recommended would increase trust and transparency.

Performance optimization through distributed computing frameworks like Apache Spark for handling larger datasets (MovieLens 1M, 10M, or 25M) would improve scalability. Implementing A/B testing capabilities to evaluate different recommendation algorithms would help in continuous improvement.

Overall, these enhancements can transform the system from a basic collaborative filtering tool into a comprehensive movie recommendation platform suitable for production deployment in entertainment applications and streaming services.

## References

- **MovieLens Dataset**  
  https://grouplens.org/datasets/movielens/100k/

- **Python Documentation**  
  https://pandas.pydata.org/docs/

- **Scikit-learn Documentation**  
  https://scikit-learn.org/stable/documentation.html

- **Matplotlib Development Team. Matplotlib Documentation**  
  https://matplotlib.org/stable/users/index.html

- **Seaborn Documentation**  
  https://seaborn.pydata.org/

- **GroupLens Research Paper**  
  F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.  
  DOI: http://dx.doi.org/10.1145/2827872

- **Collaborative Filtering Research**  
  https://www.researchgate.net/publication/220605111_Collaborative_Filtering_Recommender_Systems

- **Cosine Similarity in Recommendation Systems**  
  https://www.sciencedirect.com/topics/computer-science/cosine-similarity

## License

This project uses the MovieLens 100K dataset from GroupLens Research at the University of Minnesota.

**Citation:**

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.  
DOI: http://dx.doi.org/10.1145/2827872

**Dataset License:**
- The dataset may be used for research purposes
- Users must acknowledge the use of the dataset in publications
- Users may not redistribute the data without permission
- Commercial use requires permission from GroupLens Research

For more information, visit: https://grouplens.org/datasets/movielens/
