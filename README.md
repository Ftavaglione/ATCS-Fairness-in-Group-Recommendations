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

## Usage
1. Load the dataset using `ratings = rs.load()` and merge the ratings with movie information.
2. Compute the similarity matrix using either Pearson correlation or cosine similarity.
3. Select a target user.
4. Predict ratings for the target user.
5. Recommend top movies for the target user based on the predicted ratings.

## References
- [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/latest/)
