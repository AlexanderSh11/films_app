from collections import Counter
import os
import django
from django.core.cache import cache
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myapp.settings")
django.setup()
from app.models import Movie, Favorite, MovieRating
from django.contrib.auth.models import User

class MovieRecommender:
    def __init__(self):
        self.user_similarity = None
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.n_users = 0
        self.n_movies = 0

    def train(self):
        train_data, test_data = self.split_data(test_size=0.2)
        self.train_data_matrix = np.zeros((self.n_users, self.n_movies))
        for line in train_data.itertuples():
            self.train_data_matrix[line.userIndex, line.movieIndex] = line.rating

        self.test_data_matrix = np.zeros((self.n_users, self.n_movies))
        for line in test_data.itertuples():
            self.test_data_matrix[line.userIndex, line.movieIndex] = line.rating

        # считаем косинусное расстояние для пользователей
        self.user_similarity = pairwise_distances(self.train_data_matrix, metric='cosine')
        return self.user_similarity

    def get_data(self):
        # получаем все оценки
        all_ratings = MovieRating.objects.select_related('movie', 'user').all()
        
        # получаем все избранные фильмы
        all_favorites = Favorite.objects.select_related('movie', 'user').all()
        
        # преобразуем оценки в DataFrame
        ratings_data = []
        for rating in all_ratings:
            ratings_data.append({
                'userId': rating.user_id,
                'movieId': rating.movie_id,
                'rating': rating.user_rating,
            })
        ratings_df = pd.DataFrame(ratings_data)

        self.n_users = self.map_id_to_index(ratings_df, 'userId', 'userIndex')
        self.n_movies = self.map_id_to_index(ratings_df, 'movieId', 'movieIndex')

        print(f'Total users: {self.n_users}\nTotal movies: {self.n_movies}')
        return ratings_df

    def split_data(self, test_size=0.2):
        ratings_df = self.get_data()
        # разделяем данные
        train_data, test_data = train_test_split(ratings_df, test_size=test_size, random_state=42)

        print(f'Train shape: {train_data.shape}')
        print(f'Test shape: {test_data.shape}')
        print(f'Train data:\n{train_data.head(10)}')
        return train_data, test_data
    
    def map_id_to_index(self, df, id_column, index_column):
        # создаём отображение userId и movieId -> индекс
        ids = df[id_column].unique()
        item_to_index = {iid: i for i, iid in enumerate(ids)}
        # заменяем ID на индексы
        df[index_column] = df[id_column].map(item_to_index)
        return len(ids)
    
    def predict_user_based(self, top=5):
        try:
            # Простое усреднение оценок top наиболее похожих пользователей
            top_similar_ratings = np.zeros((self.n_users, top, self.n_movies))

            for i in range(self.n_users):
                # сортируем по возрастанию косинусного расстояния
                top_sim_users = self.user_similarity[i].argsort()[1:top + 1]
                top_similar_ratings[i] = self.train_data_matrix[top_sim_users]

            pred = np.zeros((self.n_users, self.n_movies))
            for i in range(self.n_users):
                pred[i] = top_similar_ratings[i].sum(axis=0) / top

            return pred
        except ValueError:
            print(f'Параметр top={top} выходит за пределы top_similar_ratings.shape[0]={top_similar_ratings.shape[0]}')
            return False
    
    def rmse(self, pred, actual):
        # Корень из среднеквадратичной ошибки
        mask = actual > 0
        mse = ((pred[mask] - actual[mask]) ** 2).mean()
        return np.sqrt(mse)

    def recommend_movies(self, user, n=5):
        return

def main():
    user = User.objects.first()
    if user:
        recommender = MovieRecommender()
        recommender.train()
        user_pred = recommender.predict_user_based(top=2)
        print('User similarity matrix:')
        print(recommender.user_similarity)
        if isinstance(user_pred, bool):
            return
        print('UserToUser RMSE:', recommender.rmse(user_pred, recommender.test_data_matrix))

if __name__ == "__main__":
    main()