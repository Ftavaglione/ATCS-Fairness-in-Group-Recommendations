#devo ricordarmi quali film ho già suggerito per il gruppo (10 film per ogni iterazione della sequenza)                                             check
#devo ricordarmi il grado di soddisfazione di ogni utente del gruppo
#devo ricalcolarmi ad ogni iterazione della sequenza il fattore alfa, dato dal disagreement tra utenti inteso in termini di soddisfazione

import groupRecommendations as gr
import recommenderSystem as rs
import sequentialGroupRecommendations as sgr

#def get_user_satisfaction(group_ratings, ):

ratings = rs.load_dataset()

# Compute similarity matrix
#alternatively use rs.cosine_similarity_matrix(ratings))
if rs.does_correlation_matrix_exist():
        similarity_df = rs.load_correlation_matrix()
else:
        similarity_df = rs.pearson_similarity_matrix(ratings)  
    
# Select group of members g (e.g. users with ID 1,2,3)
#usare gruppo 2,5,6 per mostrare esperimenti
group_users = [2,5,6]
group_ratings, users_ratings = gr.get_group_ratings(group_users, ratings, similarity_df)
sgr.sequentialGroupRecommendation(group_ratings, group_users, users_ratings)
   
