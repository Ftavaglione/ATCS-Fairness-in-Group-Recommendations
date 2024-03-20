# FAIRNESS RECOMMENDATION
Project for the lectures on **Fairness Recommendation** from the course **Advanced Topics in Computer Science** (ATCS)

## Repo structure
- The `/src` folder contains the source code 
- The `/ml-latest-small` folder contains the [Movielens](https://grouplens.org/datasets/movielens/) (recommended for education and development) dataset files containg 100k rows.

# Collaborative Filtering Movie Recommendation System

This repository contains code for a collaborative filtering movie recommendation system implemented in Python. The recommendation system is based on user ratings and employs two similarity metrics: Pearson correlation and cosine similarity. The system predicts movie ratings for a target user and recommends the top movies based on these predictions.

## Dataset
The dataset used in this project is the MovieLens Latest Small dataset, which consists of user ratings and movie metadata. The ratings dataset contains user ratings for various movies, while the movies dataset contains information about the movies such as titles and genres.

## Requirements
To run the code in this repository, you need to have the following dependencies installed:
- pandas
- numpy
- scikit-learn

You can install these dependencies by running:
```
pip install -r requirements.txt
```

## Functions

### 1. Pearson Similarity Matrix
- `pearson_similarity_matrix(ratings)`: Computes the Pearson correlation matrix between users based on their ratings.

### 2. Cosine Similarity Matrix
- `cosine_similarity_matrix(ratings)`: Computes the cosine similarity matrix between users based on their ratings.

### 3. Get Top Similar Users
- `get_top_similar_users(similarity_df, target_user, n=10)`: Retrieves the top similar users for a target user based on a similarity matrix.

### 4. Get User Ratings
- `get_user_ratings(ratings, user)`: Retrieves ratings of a specific user from the ratings DataFrame.

### 5. Predict Ratings
- `predict_ratings(ratings, similarity_df, target_user)`: Predicts movie ratings for a target user based on collaborative filtering.

### 6. Recommend Movies
- `recommend_movies(predictions, n=10)`: Recommends top movies for a user based on predicted ratings.

## Usage
1. Load the dataset using `ratings = rs.load()` and merge the ratings with movie information.
2. Compute the similarity matrix using either Pearson correlation or cosine similarity.
3. Select a target user.
4. Predict ratings for the target user.
5. Recommend top movies for the target user based on the predicted ratings.

## Example
```python
# Load dataset
ratings = rs.load()

# Compute similarity matrix
similarity_df = rs.pearson_similarity_matrix(ratings)

# Select target user
target_user = 1

# Predict ratings for target user
predictions = rs.predict_ratings(ratings, similarity_df, target_user)

# Recommend top movies for target user
recommended_movies = rs.recommend_movies(predictions)

# Print recommended movies
print("Top 10 recommended movies for target user with ID:", target_user)
for movie_title, predicted_rating in recommended_movies:
    print("Movie title:", movie_title, "- Predicted Rating:", round(predicted_rating, 2))
```

## References
- [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/latest/)


If you encounter any issues or have suggestions for improvements, please open an issue in the repository.
