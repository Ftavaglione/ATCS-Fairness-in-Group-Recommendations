import recommenderSystem as rs

# Load dataset
ratings = rs.load_dataset()

# Print the first rows
print(ratings.head())
print("Total number of rows:", len(ratings),"\n")

# Compute similarity matrix
#alternatively use rs.cosine_similarity_matrix(ratings)
if rs.does_correlation_matrix_exist():
        similarity_df = rs.load_correlation_matrix()
else:
        similarity_df = rs.pearson_similarity_matrix(ratings)       

# Select target user (e.g. user with ID 1)
target_user = 1
    
# Predict ratings for target user
predictions = rs.predict_ratings(ratings, similarity_df, target_user)
    
# Recommend top movies for target user
recommended_movies = rs.recommend_movies(predictions)
    
print("\nTop 10 recommended movies for target user with ID", target_user)
for movie_title, predicted_rating in recommended_movies:
        print(movie_title, "- Predicted Rating:", round(predicted_rating, 2))
