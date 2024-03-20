import itertools
import recommenderSystem as rs
import itertools

def get_group_ratings(group_users, ratings, similarity_df):
    """
    Computes ratings for movies recommended to a group of users using a recommender system.

    Args:
        group_users (list): List of user IDs in the group.
        ratings (DataFrame): DataFrame containing user ratings.
        similarity_df (DataFrame): DataFrame containing similarity scores between users.

    Returns:
        dict: A dictionary containing movie titles as keys and a list of ratings by group members as values.
    """
    ...
    group_ratings = {}
    movie_counts = {}

    for target_user in group_users:
        rated_movies = ratings[ratings['userId'] == target_user][['title', 'rating']]
        rated_movies = [(row['title'], row['rating']) for index, row in rated_movies.iterrows()]
        # Predict ratings for target user
        predictions = rs.predict_ratings(ratings, similarity_df, target_user)

        all_movies = []   
        # Combine rated and recommended movies
        all_movies += [(movie, rating) for movie, rating in predictions.items() if movie not in [title for title, _ in rated_movies]]
        all_movies += rated_movies

        for movie, rating in all_movies:
            if movie not in group_ratings:
                group_ratings[movie] = []  # Inizializziamo una lista vuota per ogni film
            group_ratings[movie].append(rating)

    movies_to_remove = []
    for movie, ratings in group_ratings.items():   
        if len(ratings) < len(group_users):
            movies_to_remove.append(movie)

    for movie in movies_to_remove:
        del group_ratings[movie]

    return group_ratings

def average_method(group_ratings, n=10):
    """
    Computes top recommended movies for a group of users using the average method.

    Args:
        group_ratings (dict): Dictionary containing movie titles as keys and a list of ratings by group members as values.
        n (int, optional): Number of top recommended movies to display. Defaults to 10.

    Returns:
        None
    """
    average_ratings = {}

    # Calcoliamo il voto medio per ciascun film
    for movie, ratings in group_ratings.items():
        average_rating = sum(ratings) / len(ratings)
        average_ratings[movie] = average_rating

    sorted_movies = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the average method:")
    for movie_title, rating in top_n_movies:
        print("Movie title:", movie_title, "- Predicted Rating:", round(rating, 2))

def least_misery_method(group_ratings, n=10):
    """
    Computes top recommended movies for a group of users using the least misery method.

    Args:
        group_ratings (dict): Dictionary containing movie titles as keys and a list of ratings by group members as values.
        n (int, optional): Number of top recommended movies to display. Defaults to 10.

    Returns:
        None
    """
    min_ratings = {}

    for movie, ratings in group_ratings.items():
        min_rating = min(ratings)
        min_ratings[movie] = min_rating

    sorted_movies = sorted(min_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the least misery method:")
    for movie_title, rating in top_n_movies:
        print("Movie title:", movie_title, "- Predicted Rating:", round(rating, 2))

def pairwise_disagreement(filtered_df, movie_id, group):
    """
    Computes pairwise disagreement among group members for a specific movie.

    Args:
        filtered_df (DataFrame): DataFrame containing filtered ratings.
        movie_id (int): ID of the movie.
        group (list): List of user IDs in the group.

    Returns:
        float: Pairwise disagreement value.
    """
    
    num_members = len(group)

    abs_sum = 0

    for user_a,user_b in itertools.combinations(group,2):
        if user_a != user_b:
            rating_a = filtered_df.loc[(filtered_df['userId'] == user_a) & (filtered_df['movieId'] == movie_id), 'rating'].values[0]
            rating_b = filtered_df.loc[(filtered_df['userId'] == user_b) & (filtered_df['movieId'] == movie_id), 'rating'].values[0]
            abs_sum += abs(rating_a-rating_b)

    return (2/(num_members*(num_members-1)))*abs_sum

def average_pairwise_disagreement(filtered_df, movie_id, group, w):
    """
    Computes the average pairwise disagreement weighted by the average rating for a movie.

    Args:
        filtered_df (DataFrame): DataFrame containing filtered ratings.
        movie_id (int): ID of the movie.
        group (list): List of user IDs in the group.
        w (float): Weight parameter for disagreement.

    Returns:
        float: Weighted average pairwise disagreement.
    """
    
    movie_ratings = filtered_df[filtered_df['movieId'] == movie_id]['rating']
    
    average_rating = movie_ratings.mean() #Calculate the average
    
    disagreement_value = pairwise_disagreement(filtered_df,movie_id,group)
    
    return ((1-w) * average_rating)+(w * disagreement_value)