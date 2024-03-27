import groupRecommendations as gr

def sequentialGroupRecommendation(group_ratings, group, users_ratings, j=3):
    users_sat_prev = {}
    
    for i in range(j):
        print("**  users_sat_prev:",users_sat_prev)
        group_ratings, users_sat_prev = gr.hybrid_aggregration_method(group_ratings, users_sat_prev, group, users_ratings, i)