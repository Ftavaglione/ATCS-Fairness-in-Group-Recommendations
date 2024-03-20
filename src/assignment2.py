import groupRecommendations as gr
import recommenderSystem as rs

# Load dataset
ratings = rs.load()

#similarity_df = cosine_similarity_matrix(ratings)
similarity_df = rs.pearson_similarity_matrix(ratings)
    
# Select group of members g (e.g. users with ID 1,2,3)
group_users = [1,2,3]
group_ratings = gr.get_group_ratings(group_users, ratings, similarity_df)

gr.average_method(group_ratings)
gr.least_misery_method(group_ratings)
