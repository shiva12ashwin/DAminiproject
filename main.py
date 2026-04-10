from recommender import get_all_movies, get_user_ids, recommend_by_movie, recommend_for_user


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
            try:
                results = recommend_by_movie(int(movie_id))
                print(f"\nMovies similar to movie ID {movie_id}:")
                for i, movie in enumerate(results, 1):
                    print(f"{i}. {movie['title']} (Score: {movie['score']:.4f})")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            user_id = input("Enter user ID: ").strip()
            try:
                results = recommend_for_user(int(user_id))
                print(f"\nRecommendations for user {user_id}:")
                for i, movie in enumerate(results, 1):
                    print(f"{i}. {movie['title']} (Score: {movie['score']:.4f})")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            movies = get_all_movies()
            print(f"\nTotal movies: {len(movies)}")
            print("First 20 movies:")
            for movie in movies[:20]:
                print(f"ID {movie['movie_id']}: {movie['title']}")
            print("...")
        
        elif choice == "4":
            users = get_user_ids()
            print(f"\nTotal users: {len(users)}")
            print(f"User IDs range from {min(users)} to {max(users)}")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
