import groupRecommendations as gr

def sequentialGroupRecommendation(group_ratings, group, users_ratings, j=3):
    """
    Perform sequential group recommendation for a given number of iterations.

    Args:
        group_ratings (dict): Dictionary of movie ratings for the group.
        group (list): List of user IDs in the group.
        users_ratings (dict): Dictionary containing users' ratings.
        j (int, optional): Number of iterations. Defaults to 3.
    """
    users_sat_prev = {}
    
    for i in range(j):
        print("**  users_sat_prev:",users_sat_prev)
        group_ratings, users_sat_prev = gr.hybrid_aggregration_method(group_ratings, users_sat_prev, group, users_ratings, i)