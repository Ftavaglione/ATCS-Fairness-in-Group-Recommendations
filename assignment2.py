import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import time

# Compute Pearson correlation between users
def pearson_similarity_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    correlation_matrix = np.corrcoef(user_movie_matrix)
    np.fill_diagonal(correlation_matrix, 0)  # Setting self-similarity to 0
    correlation_df = pd.DataFrame(correlation_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return correlation_df

# Compute cosine similarity between users
def cosine_similarity_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
   
    similarity_matrix = cosine_similarity(user_movie_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Setting self-similarity to 0
    similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    
    return similarity_df

# Get top similar users based on similarity
def get_top_similar_users(similarity_df, target_user, n=10):
    similar_users = similarity_df[target_user].sort_values(ascending=False)[0:n]
    print("Top 10 most similar users for target user with ID:",target_user,"\n",similar_users.index,"\n")
    return similar_users

def get_user_ratings(ratings, user):
    return ratings[ratings['userId'] == user]

# Predict movie ratings for target user
def predict_ratings(ratings, similarity_df, target_user):
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
def recommended_movies(predictions, rated_movies, n=10):
    all_movies = []   
    # Combine rated and recommended movies
    all_movies += [(movie, rating) for movie, rating in predictions.items() if movie not in [title for title, _ in rated_movies]]
    all_movies += rated_movies
    
    # Sort the combined list in descending order of ratings
    all_movies = sorted(all_movies, key=lambda x: x[1], reverse=True)
    # Select top N movies
    top_n_movies = all_movies[:n]

    return top_n_movies

def get_group_ratings(group_users, ratings):
    group_ratings = {}
    movie_counts = {}

    for target_user in group_users:
        rated_movies = ratings[ratings['userId'] == target_user][['title', 'rating']]
        rated_movies = [(row['title'], row['rating']) for index, row in rated_movies.iterrows()]
        # Predict ratings for target user
        predictions = predict_ratings(ratings, similarity_df, target_user)

        all_movies = []   
        # Combine rated and recommended movies
        all_movies += [(movie, rating) for movie, rating in predictions.items() if movie not in [title for title, _ in rated_movies]]
        all_movies += rated_movies
        #DA MODIFICARE

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
    min_ratings = {}

    for movie, ratings in group_ratings.items():
        min_rating = min(ratings)
        min_ratings[movie] = min_rating

    sorted_movies = sorted(min_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the least misery method:")
    for movie_title, rating in top_n_movies:
        print("Movie title:", movie_title, "- Predicted Rating:", round(rating, 2))


def disagreement_weighted_group_recommendations(ratings, similarity_df, group_users):
    group_ratings = ratings[ratings['userId'].isin(group_users)]
    group_std_rating = group_ratings.groupby('movieId')['rating'].std().fillna(0)  # Calcola la deviazione standard delle valutazioni del film nel gruppo
    group_mean_rating = group_ratings.groupby('movieId')['rating'].mean()
    
    weighted_group_ratings = group_mean_rating * (1 - group_std_rating)  # Utilizza la deviazione standard come peso inverso
    return weighted_group_ratings.sort_values(ascending=False).head(10)


# Load dataset
ratings = pd.read_csv('ratings.csv',usecols=range(3))
movies = pd.read_csv('movies.csv',usecols=range(2))
    
ratings = pd.merge(ratings, movies)

#similarity_df = cosine_similarity_matrix(ratings)
similarity_df = pearson_similarity_matrix(ratings)
    
# Select group of members g (e.g. users with ID 1,2,3)
group_users = [1,2]
group_ratings = get_group_ratings(group_users, ratings)

average_method(group_ratings)
least_misery_method(group_ratings)
