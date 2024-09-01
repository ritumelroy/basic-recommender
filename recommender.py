import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Load the MovieLens datasets
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Prepare user and movie IDs
user_ids = ratings['userId'].unique().tolist()
movie_ids = ratings['movieId'].unique().tolist()

# Map IDs to indices
user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# Map movie indices back to IDs and titles
index_to_movie = {idx: movie_id for movie_id, idx in movie_to_index.items()}
movie_id_to_title = movies.set_index('movieId')['title'].to_dict()

# Convert IDs to indices in the dataset
ratings['userId'] = ratings['userId'].map(user_to_index)
ratings['movieId'] = ratings['movieId'].map(movie_to_index)

# Split the dataset into train and test sets
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size, embeddings_initializer='he_normal')
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.movie_bias = tf.keras.layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        user_bias = self.user_bias(inputs[:, 0])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

# Set hyperparameters
num_users = len(user_to_index)
num_movies = len(movie_to_index)
embedding_size = 50  # Latent factors

model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Prepare training data
train_user_movie_pairs = train[['userId', 'movieId']].values
train_labels = train['rating'].values

# Train the model
history = model.fit(
    train_user_movie_pairs,
    train_labels,
    batch_size=64,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# Prepare test data
test_user_movie_pairs = test[['userId', 'movieId']].values
test_labels = test['rating'].values

# Evaluate the model
test_loss = model.evaluate(test_user_movie_pairs, test_labels)
print(f"Test RMSE: {tf.sqrt(test_loss).numpy()}")

# Function to make movie recommendations for a given user
def recommend_movies(user_id, num_recommendations=5):
    user_index = user_to_index[user_id]
    user_vector = tf.fill([len(movie_ids)], user_index)  # Match shape with movie_vector
    movie_vector = tf.range(len(movie_ids))
    predictions = model.predict(tf.stack([user_vector, movie_vector], axis=1))
    predicted_ratings = pd.Series(predictions[:, 0], index=movie_ids)
    
    # Map movie IDs to titles
    predicted_ratings.index = predicted_ratings.index.map(index_to_movie)
    
    # Sort by predicted ratings
    recommended_movies = predicted_ratings.sort_values(ascending=False).head(num_recommendations)
    
    # Map movie IDs to titles
    recommended_movies_titles = recommended_movies.index.map(movie_id_to_title)
    return recommended_movies_titles

user_id = 1
recommended_movies = recommend_movies(user_id, num_recommendations=5)
print("Recommended Movies for User ID 1:")
print(recommended_movies)
