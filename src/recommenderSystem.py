import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load():
    ratings = pd.read_csv('ml-latest-small/ratings.csv',usecols=range(3))
    movies = pd.read_csv('ml-latest-small/movies.csv',usecols=range(2))
    ratings = pd.merge(ratings, movies)
    return ratings

# Compute Pearson correlation between users
def pearson_similarity_matrix(ratings):
    """
    Computes the Pearson correlation matrix between users based on their ratings.

    Parameters:
    - ratings (DataFrame): DataFrame containing user ratings with columns: 'userId', 'movieId', 'rating'.

    Returns:
    - correlation_df (DataFrame): DataFrame containing the Pearson correlation matrix between users.
    """

    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    correlation_matrix = np.corrcoef(user_movie_matrix)
    np.fill_diagonal(correlation_matrix, 0)  # Setting self-similarity to 0
    correlation_df = pd.DataFrame(correlation_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return correlation_df

# Compute cosine similarity between users
def cosine_similarity_matrix(ratings):
    """
    Computes the cosine similarity matrix between users based on their ratings.

    Parameters:
    - ratings (DataFrame): DataFrame containing user ratings with columns: 'userId', 'movieId', 'rating'.

    Returns:
    - similarity_df (DataFrame): DataFrame containing the cosine similarity matrix between users.
    """
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
   
    similarity_matrix = cosine_similarity(user_movie_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Setting self-similarity to 0
    similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    
    return similarity_df

# Get top similar users based on similarity
def get_top_similar_users(similarity_df, target_user, n=40):
    """
    Retrieves the top similar users for a target user based on a similarity matrix.

    Parameters:
    - similarity_df (DataFrame): DataFrame containing the similarity matrix between users.
    - target_user (int): The ID of the target user.
    - n (int): Number of top similar users to retrieve. Default is 10.

    Returns:
    - similar_users (Series): Series containing the top similar users for the target user.
    """
    similar_users = similarity_df[target_user].sort_values(ascending=False)[0:n]
    top_similar_users = similar_users.head(10)
    print("Top 10 most similar users for target user with ID:",target_user,"\n",top_similar_users.index,"\n")
    return similar_users

def get_user_ratings(ratings, user):
    """
    Retrieves ratings of a specific user from the ratings DataFrame.

    Parameters:
    - ratings (DataFrame): DataFrame containing user ratings with columns: 'userId', 'movieId', 'rating'.
    - user (int): The ID of the user.

    Returns:
    - user_ratings (DataFrame): DataFrame containing ratings of the specified user.
    """
    return ratings[ratings['userId'] == user]

# Predict movie ratings for target user
def predict_ratings(ratings, similarity_df, target_user):
    """
    Predicts movie ratings for a target user based on collaborative filtering.

    Parameters:
    - ratings (DataFrame): DataFrame containing user ratings with columns: 'userId', 'movieId', 'rating'.
    - similarity_df (DataFrame): DataFrame containing the similarity matrix between users.
    - target_user (int): The ID of the target user.

    Returns:
    - predicted_ratings (dict): Dictionary containing predicted ratings for movies.
    """
    target_user_ratings = ratings[ratings['userId'] == target_user]
    target_user_mean_rating = target_user_ratings['rating'].mean()
    
    predicted_ratings = {}
    similar_users_mean_rating={}

    similar_users = get_top_similar_users(similarity_df, target_user)

    similar_users_seen_movies = ratings[ratings['userId'].isin(similar_users.index)]
    unseen_movies = similar_users_seen_movies[~similar_users_seen_movies['movieId'].isin(target_user_ratings['movieId'])][['movieId', 'title']].drop_duplicates() 

    for user, similarity in similar_users.items():
        user_ratings = get_user_ratings(ratings, user)
        similar_users_mean_rating[user] = user_ratings['rating'].mean()

    for _, row in unseen_movies.iterrows():
        
        movie_id = row['movieId']
        movie_title = row['title']
        
        weighted_sum = 0
        similarity_sum = 0
        
        for user, similarity in similar_users.items():
            
            user_ratings = get_user_ratings(ratings, user)
            
            user_rating = user_ratings[user_ratings['movieId'] == movie_id]['rating']
            
            if not user_rating.empty:
                user_rating = user_rating.values[0]
                weighted_sum += similarity * (user_rating - similar_users_mean_rating[user])
                similarity_sum += similarity
                
        if similarity_sum != 0:
            predicted_rating = target_user_mean_rating + (weighted_sum / similarity_sum)
            predicted_ratings[movie_title] = predicted_rating      

    return predicted_ratings

# Recommend top movies for target user
def recommend_movies(predictions, n=10):
    """
    Recommends top movies for a user based on predicted ratings.

    Parameters:
    - predictions (dict): Dictionary containing predicted ratings for movies.
    - n (int): Number of top movies to recommend. Default is 10.

    Returns:
    - top_movies (list): List of tuples containing top recommended movies and their predicted ratings.
    """
    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_movies