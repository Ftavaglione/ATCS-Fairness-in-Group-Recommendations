import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Predict movie ratings for target user
def predict_ratings(ratings, similarity_df, target_user):
    target_user_ratings = ratings[ratings['userId'] == target_user]
    target_user_mean_rating = target_user_ratings['rating'].mean()
    
    predicted_ratings = {}

    unseen_movies = ratings[~ratings['movieId'].isin(target_user_ratings['movieId'])][['movieId', 'title']].drop_duplicates()
    similar_users = get_top_similar_users(similarity_df, target_user)
    
    for index, row in unseen_movies.iterrows():
        movie_id = row['movieId']
        movie_title = row['title']
        
        weighted_sum = 0
        similarity_sum = 0
        
        for user, similarity in similar_users.items():
            user_ratings = ratings[ratings['userId'] == user]
            user_mean_rating = user_ratings['rating'].mean()
            user_rating = user_ratings[user_ratings['movieId'] == movie_id]['rating']
            
            if not user_rating.empty:
                user_rating = user_rating.values[0]
                weighted_sum += similarity * (user_rating - user_mean_rating)
                similarity_sum += similarity
                
        if similarity_sum != 0:
            predicted_rating = target_user_mean_rating + (weighted_sum / similarity_sum)
            predicted_ratings[movie_title] = predicted_rating
            
    return predicted_ratings

# Recommend top movies for target user
#def recommend_movies(predictions, n=10):
  #  top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
  #  return top_movies

# Recommend top movies for target user
def recommend_movies(predictions, rated_movies, n=10):
    all_movies = []   
    # Combine rated and recommended movies
    all_movies += [(movie, rating) for movie, rating in predictions.items() if movie not in [title for title, _ in rated_movies]]
    all_movies += rated_movies
    
    # Sort the combined list in descending order of ratings
    all_movies = sorted(all_movies, key=lambda x: x[1], reverse=True)
    
    # Select top N movies
    top_n_movies = all_movies[:n]

    return top_n_movies


# Load dataset
ratings = pd.read_csv('ratings.csv',usecols=range(3))
movies = pd.read_csv('movies.csv',usecols=range(2))
    
ratings = pd.merge(ratings, movies)

#similarity_df = cosine_similarity_matrix(ratings)
similarity_df = pearson_similarity_matrix(ratings)
    
# Select group g members (e.g. users with ID 1,2,3)
target_user = 1

rated_movies = ratings[ratings['userId'] == target_user][['title', 'rating']]
rated_movies = [(row['title'], row['rating']) for index, row in rated_movies.iterrows()]
    
# Predict ratings for target user
predictions = predict_ratings(ratings, similarity_df, target_user)
    
# Recommend top movies for target user
recommended_movies = recommend_movies(predictions, rated_movies)
    
print("Top 10 recommended movies for target user with ID:", target_user)
for movie_title, predicted_rating in recommended_movies:
        print("Movie title:", movie_title, "- Predicted Rating:", round(predicted_rating, 2))
