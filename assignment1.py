import pandas as pd

def get_most_similar_users(user_id, corr_matrix, n=5):
    user_corr = corr_matrix[user_id]
    similar_users = user_corr.sort_values(ascending=False).drop(user_id).head(n).index
    return similar_users

# Open the CSV file
df = pd.read_csv('ratings.csv')   
# Print the first rows
print(df.head())
print("Total number of rows:", len(df))

user_ratings = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

print(user_ratings)

# Compute Pearson correlation matrix
#corr_matrix = user_ratings.corr()

similar_users = get_top_similar_users(1, corr_matrix)

print("Top 5 similar users to user", 1, ":", similar_users)



