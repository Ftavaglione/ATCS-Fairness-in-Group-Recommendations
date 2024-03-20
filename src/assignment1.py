import recommenderSystem as rs

# Load dataset
ratings = rs.load()

# Print the first rows
print(ratings.head())
print("Total number of rows:", len(ratings),"\n")

# Compute similarity matrix
#similarity_df = cosine_similarity_matrix(ratings)
similarity_df = rs.pearson_similarity_matrix(ratings)
    
# Select target user (e.g. user with ID 1)
target_user = 1
    
# Predict ratings for target user
predictions = rs.predict_ratings(ratings, similarity_df, target_user)
    
# Recommend top movies for target user
recommended_movies = rs.recommend_movies(predictions)
    
print("Top 10 recommended movies for target user with ID: ", target_user)
for movie_title, predicted_rating in recommended_movies:
        print("Movie title:", movie_title, "- Predicted Rating:", round(predicted_rating, 2))
