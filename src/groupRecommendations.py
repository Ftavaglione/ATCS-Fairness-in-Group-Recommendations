import itertools
import recommenderSystem as rs
import numpy as np

def get_user_rating(user_rating_list, target_user_id):
    """
    Retrieves the rating of a specific user from a list of user ratings.

    Args:
        user_rating_list (list): List of tuples (user_id, rating).
        target_user_id (int): ID of the target user.

    Returns:
        float or None: Rating of the target user if found, None otherwise.
    """
    for user_id, rating in user_rating_list:
        # Check if the current user_id matches the target_user
        if user_id == target_user_id:
            # Return the rating if found
            return rating
    return None

def get_group_ratings(group_users, ratings, similarity_df):
    """
    Computes ratings for movies recommended to a group of users using a recommender system.

    Args:
        group_users (list): List of user IDs in the group.
        ratings_df (DataFrame): DataFrame containing user ratings.
        similarity_df (DataFrame): DataFrame containing similarity scores between users.

    Returns:
        dict: A dictionary containing movie titles as keys and a list of tuples (user_id, rating) by group members.
    """
    group_ratings = {}
    users_ratings = {}

    for target_user in group_users:

        if target_user not in users_ratings:
            users_ratings[target_user] = {}

        # Predict ratings for target user
        predictions = rs.predict_ratings(ratings, similarity_df, target_user)

        users_ratings[target_user] = predictions

        # Combine rated and recommended movies
        all_movies = [(movie, rating) for movie, rating in predictions.items()]

        for movie, rating in all_movies:
            if movie not in group_ratings:
                group_ratings[movie] = []  
            group_ratings[movie].append((target_user, rating))

    movies_to_remove = []
    #remove the films that have not been predicted for all users in the group
    for movie, ratings in group_ratings.items():   
        if len(ratings) < len(group_users):
            movies_to_remove.append(movie)

    for movie in movies_to_remove:
        del group_ratings[movie]

    return group_ratings, users_ratings

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

    for movie, ratings in group_ratings.items():
        ratings_only = [rating for _, rating in ratings]
        average_rating = sum(ratings_only) / len(ratings_only)
        average_ratings[movie] = average_rating

    sorted_movies = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the average method:")
    for movie_title, rating in top_n_movies:
        print(movie_title, "- Predicted Rating:", round(rating, 2))

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
        ratings_only = [rating for _, rating in ratings]
        min_rating = min(ratings_only)
        min_ratings[movie] = min_rating

    sorted_movies = sorted(min_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the least misery method:")
    for movie_title, rating in top_n_movies:
        print(movie_title, "- Predicted Rating:", round(rating, 2))

def pairwise_disagreement(user_rating_list, group):
    """
    Computes pairwise disagreement among group members for a specific movie.

    Args:
        user_rating_list (list): List of tuples (user_id, rating).
        group (list): List of user IDs in the group.

    Returns:
        float: Pairwise disagreement value.
    """
    num_members = len(group)
    abs_sum = 0

    for user_a,user_b in itertools.combinations(group,2):
        if user_a != user_b:
            rating_a = get_user_rating(user_rating_list, user_a)
            rating_b = get_user_rating(user_rating_list, user_b)

            abs_sum += abs(rating_a-rating_b)
    
    return (2/(num_members*(num_members-1)))*abs_sum

def average_pairwise_disagreement(user_rating_list, group, w=0.3):
    """
    Computes the average pairwise disagreement weighted by the average rating for a movie.

    Args:
        user_rating_list (list): List of tuples (user_id, rating).
        group (list): List of user IDs in the group.
        w (float): Weight parameter for disagreement.

    Returns:
        float: Weighted average pairwise disagreement.
    """
    ratings_only = [rating for _, rating in user_rating_list]
    average_rating = np.mean(ratings_only)

    disagreement_value = pairwise_disagreement(user_rating_list,group)
    
    return ((1-w) * average_rating)+(w * disagreement_value)

def pairwise_disagreement_method(group_ratings, group, n=10):
    pairwise_disagreement_ratings = {}

    for movie, user_rating_list in group_ratings.items():
        pairwise_disagreement_ratings[movie] = average_pairwise_disagreement(user_rating_list, group)

    sorted_movies = sorted(pairwise_disagreement_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_movies[:n]

    print("\nTop 10 recommended movies for group users using the pairwise disagreement method:")
    for movie_title, rating in top_n_movies:
         print(movie_title, "- Predicted Rating:", round(rating, 2))

def hybrid_aggregration_method(group_ratings, users_sat_prev, group, users_recommend_ratings, i, a=0, n=10):
    """
    Calculate top recommended movies for a group using the hybrid method.

    Args:
        group_ratings (dict): Dictionary of movie ratings for the group.
        users_sat_prev (dict): Dictionary containing previous satisfaction levels of users.
        group (list): List of user IDs in the group.
        users_recommend_ratings (dict): Dictionary containing recommended ratings for users.
        i (int): Iteration number.
        a (float, optional): Weight parameter. Defaults to 0.
        n (int, optional): Number of top movies to recommend. Defaults to 10.

    Returns:
        tuple: A tuple containing updated group ratings and users' satisfaction levels.
    """
    users_ratings = {}
    users_sat = {}
    hybrid_ratings = {}

    #update dynamically a based on disagreements on previous iteration 
    if i>0:
        a = hybrid_pairwise_disagreement(users_sat_prev, group)
        print("\na has been re-calculated:",round(a,2))

    for movie, ratings in group_ratings.items():
         
         ratings_only = [rating for _, rating in ratings]

         average_rating = sum(ratings_only) / len(ratings_only)
         min_rating = min(ratings_only)

         hybrid_ratings[movie] = (1-a) * average_rating + a * min_rating

    sorted_movies = sorted(hybrid_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_group_movies = sorted_movies[:n]

    for movie, rating in top_n_group_movies:
        for user in group:
              if user not in users_ratings:
                users_ratings[user] = {} 

              users_ratings[user][movie] = get_user_rating(group_ratings[movie], user)

    for user in group:
        recommendListSat = 0
        groupListSat = 0
    
        for movie_title, rating in users_ratings[user].items():
            groupListSat += rating

        # must obtain top 10 movies recommended to the user
        top_n_user_movies = rs.recommend_movies(users_recommend_ratings[user])

        for movie, rating in top_n_user_movies:
            recommendListSat += rating

        if user not in users_sat:
                users_sat[user] = 0 

        userSat = groupListSat / recommendListSat
        #abbiamo un dizionario in cui memorizziamo il gradimento di ciascun utente nel gruppo 
        users_sat[user] = userSat
        print("user",user,"satisfaction is:",users_sat[user])

    print("\nTop 10 recommended movies for group users using the hybrid method at iteration ",i+1,":")
    for movie_title, rating in top_n_group_movies:
         del group_ratings[movie_title]
         print(movie_title, "- Predicted Rating:", round(rating, 2))

    return group_ratings, users_sat

def hybrid_pairwise_disagreement(users_sat_prev, group):
    """
    Calculate pairwise disagreement among users' satisfaction levels.

    Args:
        users_sat_prev (dict): Dictionary containing previous satisfaction levels of users.
        group (list): List of user IDs in the group.

    Returns:
        float: Pairwise disagreement value.
    """
    num_members = len(group)
    abs_sum = 0

    for user_a,user_b in itertools.combinations(group,2):
        if user_a != user_b:
            satisfaction_a = users_sat_prev[user_a]
            satisfaction_b = users_sat_prev[user_b]

            abs_sum += abs(satisfaction_a - satisfaction_b)
    
    return (2/(num_members*(num_members-1)))*abs_sum



