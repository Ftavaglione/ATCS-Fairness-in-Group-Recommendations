# FAIRNESS RECOMMENDATION
Project for the lectures on **Fairness Recommendation** from the course **Advanced Topics in Computer Science** (ATCS)

## Repo structure
- The `/src` folder contains the source code
- The `/ml-latest-small` folder contains the [Movielens](https://grouplens.org/datasets/movielens/) (recommended for education and development) dataset files containg 100k rows.

## Assignments
### Assignment 1: User-based Collaborative Filtering Recommendations

The goal of the first assignment is to implement a user-based collaborative filtering
approach.

- **(a)** Download the MovieLens 100K rating dataset from https://grouplens.org/datasets/
movielens/ (the small dataset recommended for education and development). Read the
dataset, display the first few rows to understand it, and display the count of ratings (rows)
in the dataset to be sure that you download it correctly.

- **(b)** Implement the user-based collaborative filtering approach, using the Pearson
correlation function for computing similarities between users, and
**(c)** the prediction function presented in class for predicting movies scores.

- **(d)** Select a user from the dataset, and for this user, show the 10 most similar users and the 10 most relevant movies that the recommender suggests.

- **(e)** Design and implement a new similarity function for computing similarities between
users. Explain why this similarity function is useful for the collaborative filtering approach.
Hint: Exploiting ideas from related works are highly encouraged.

### Assignment 2: Group Recommendations
The goal of the second assignment is to implement existing approaches for producing
group recommendations and propose your own ideas and implement them for the same
topic.

- **(a)** For producing group recommendation, we will use the user-based collaborative
filtering approach as this implemented in Assignment 1. Specifically, for producing group
recommendations, we will first compute the movies recommendations for each user in
the group, and then we will aggregate the lists of the individual users, so as to produce a
single list of movies for the group.
    
    You will implement two well established aggregation methods for producing the group recommendations.

    The first aggregation approach is the average method. The main idea behind this approach is that all members are considered equals. So, the rating of an item for a group of users will be given be averaging the scores of an item across all group members.

    The second aggregation method is the least misery method, where one member can act as a veto for the rest of the group. In this case, the rating of an item for a group of users is computed as the minimum score assigned to that item in all group members recommendations.

    Produce a group of 3 users, and for this group, show the top-10 recommendations, i.e., the 10 movies with the highest prediction scores that:
    - the average method suggests
    - the least misery method suggest.


- **(b)** The methods employed in part (a) of Assignment 2, do not consider any
disagreements between the users in the group.
In part (b) of Assignment 2, define a way for counting the disagreements between the
users in a group, and propose a method that takes disagreements into account when
computing suggestions for the group.
Implement your method and explain why it is useful when producing group
recommendations.
Use again the group of 3 users, and for this group, show the top-10 recommendations,
i.e., the 10 movies with the highest prediction scores that your method suggests. Use the
MovieLens 100K rating dataset.

### Assignment 3: Sequential Recommendations
Motivated by the sequential methods we discussed in class, the goal of the third
assignment is to design and implement a new method for producing sequential group
recommendations. Also, provide detailed explanations and clarifications about why the
method you propose works well for the case of sequential group recommendations.

Produce a group of 3 users, and for this group, show the top-10 recommendations in 3
different sequences, i.e., the 10 movies with the highest prediction scores in 3 rounds.
