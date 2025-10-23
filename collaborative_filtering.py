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
        self.user_to_index = None
        self.movie_to_index = None
        self.index_to_user = None
        self.index_to_movie = None

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

        ratings_data = []

        # Создаем множество для быстрой проверки существующих оценок
        existing_ratings_set = set()
        
        # Добавляем реальные оценки
        for rating in all_ratings:
            user_movie_key = (rating.user_id, rating.movie_id)
            existing_ratings_set.add(user_movie_key)
            ratings_data.append({
                'userId': rating.user_id,
                'movieId': rating.movie_id,
                'rating': rating.user_rating,
            })
        
        # Добавляем избранные как неявные оценки (rating=9)
        for favorite in all_favorites:
            user_movie_key = (favorite.user_id, favorite.movie_id)
            
            # Быстрая проверка через множество O(1)
            if user_movie_key not in existing_ratings_set:
                ratings_data.append({
                    'userId': favorite.user_id,
                    'movieId': favorite.movie_id,
                    'rating': 9,
                })
                # Добавляем в множество чтобы избежать дубликатов
                existing_ratings_set.add(user_movie_key)

        # преобразуем оценки в DataFrame
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
        # сохраняем mapping для обратного преобразования
        if id_column == 'userId':
            self.user_to_index = item_to_index
            self.index_to_user = {v: k for k, v in item_to_index.items()}
        elif id_column == 'movieId':
            self.movie_to_index = item_to_index
            self.index_to_movie = {v: k for k, v in item_to_index.items()}
        # заменяем ID на индексы
        df[index_column] = df[id_column].map(item_to_index)
        return len(ids)
    
    def predict_user_based_naive(self, top=5):
        """
        Используется простой подход без учета средних оценок
        Нет нормализации данных
        Не учитываются смещения пользователей
        """
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
        
    def predict_user_based_k_fract_mean(self, top=5):
        """
        1. Учитываются средние оценки каждого пользователя
        2. Производится нормализация оценок (вычитание среднего)
        3. Используется взвешенное среднее на основе схожести пользователей
        4. Учитываются только реальные оценки (> 0)
        
        Формула: pred_ui = mean_u + sum(sim(u,v) * (r_vi - mean_v)) / sum|sim(u,v)|
        где:
        - pred_ui - предсказанная оценка пользователя u для фильма i
        - mean_u - средняя оценка пользователя u
        - sim(u,v) - схожесть пользователей u и v
        - r_vi - оценка пользователя v для фильма i
        - mean_v - средняя оценка пользователя v
        """
        try:
            # Вычисляем средние оценки для каждого пользователя
            user_mean_ratings = np.zeros(self.n_users)
            for i in range(self.n_users):
                user_ratings = self.train_data_matrix[i]
                rated_movies = user_ratings > 0  # только ненулевые оценки
                if np.any(rated_movies):
                    user_mean_ratings[i] = np.mean(user_ratings[rated_movies])
                else:
                    user_mean_ratings[i] = 0  # если нет оценок

            pred = np.zeros((self.n_users, self.n_movies))
            
            for i in range(self.n_users):  # для каждого пользователя
                # Находим top наиболее похожих пользователей (исключая самого себя)
                # Используем [1:top+1] чтобы пропустить самого пользователя (схожесть = 1)
                similar_users_idx = self.user_similarity[i].argsort()[1:top + 1]
                
                # Получаем схожести с этими пользователями
                similarities = self.user_similarity[i][similar_users_idx]
                
                # Средняя оценка текущего пользователя
                current_user_mean = user_mean_ratings[i]
                
                # Временные массивы для вычислений
                weighted_sum = np.zeros(self.n_movies)
                similarity_sum = np.zeros(self.n_movies)
                
                for j, sim_user_idx in enumerate(similar_users_idx):
                    similarity = similarities[j]
                    
                    # Оценки похожего пользователя
                    similar_user_ratings = self.train_data_matrix[sim_user_idx]
                    
                    # Средняя оценка похожего пользователя
                    similar_user_mean = user_mean_ratings[sim_user_idx]
                    
                    # Нормализованные оценки (вычитаем среднее)
                    normalized_ratings = similar_user_ratings - similar_user_mean
                    
                    # Маска только для реальных оценок
                    valid_ratings_mask = similar_user_ratings > 0
                    
                    # Взвешенная сумма с учетом схожести
                    weighted_sum[valid_ratings_mask] += similarity * normalized_ratings[valid_ratings_mask]
                    similarity_sum[valid_ratings_mask] += np.abs(similarity)
                
                # Вычисляем предсказания
                for movie_idx in range(self.n_movies):
                    if similarity_sum[movie_idx] > 0:
                        # pred = mean_u + sum(sim * (r - mean_v)) / sum|sim|
                        pred[i, movie_idx] = current_user_mean + weighted_sum[movie_idx] / similarity_sum[movie_idx]
                    else:
                        # Если нет похожих пользователей, оценивших этот фильм, используем среднее пользователя
                        pred[i, movie_idx] = current_user_mean
                
                # Ограничиваем предсказания разумными пределами
                pred[i] = np.clip(pred[i], 0, 10)
            
            return pred
            
        except Exception as e:
            print(f'Ошибка в predict_user_based_k_fract_mean: {e}')
            return False
    
    def rmse(self, pred, actual):
        # Корень из среднеквадратичной ошибки
        mask = actual > 0
        mse = ((pred[mask] - actual[mask]) ** 2).mean()
        return np.sqrt(mse)
    
    def precision_recall_at_k(self, pred, actual, k=10, threshold=3.5):
        """
        Precision - доля релевантных среди top-K рекомендаций
        Recall - доля найденных релевантных из всех релевантных
        """
        precision_list = []
        recall_list = []
        
        for i in range(self.n_users):
            # Реальные положительные оценки пользователя
            actual_positive = set(np.where(actual[i] >= threshold)[0])
            
            # Предсказанные top-K фильмов
            predicted_indices = np.argsort(pred[i])[-k:][::-1]
            predicted_set = set(predicted_indices)
            
            # Вычисляем precision и recall
            if len(predicted_set) > 0:
                precision = len(actual_positive & predicted_set) / len(predicted_set)
            else:
                precision = 0
                
            if len(actual_positive) > 0:
                recall = len(actual_positive & predicted_set) / len(actual_positive)
            else:
                recall = 0
                
            precision_list.append(precision)
            recall_list.append(recall)
        
        return np.mean(precision_list), np.mean(recall_list)

    def f1_score(self, pred, actual, k=10, threshold=3.5):
        # Гармоническое среднее precision и recall
        precision, recall = self.precision_recall_at_k(pred, actual, k, threshold)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def recommend_movies(self, user_id, pred, n=10, include_ratings=False):
        """
        Получить рекомендуемые фильмы для конкретного пользователя
        user_id: ID пользователя
        pred: матрица предсказаний
        n: количество рекомендаций
        include_ratings: включать ли предсказанные оценки
        """
        if self.user_to_index is None or self.index_to_movie is None:
            print("Ошибка: mapping не инициализирован")
            return []
        
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            print(f"Пользователь с ID {user_id} не найден в данных")
            return []
        
        # Получаем предсказания для пользователя
        user_predictions = pred[user_index]
        
        # Получаем фильмы, которые пользователь уже оценил
        user_rated_movies = set(np.where(self.train_data_matrix[user_index] > 0)[0])
        
        # Сортируем фильмы по предсказанной оценке (исключая уже оцененные)
        movie_scores = []
        for movie_index, score in enumerate(user_predictions):
            if movie_index not in user_rated_movies:
                movie_scores.append((movie_index, score))
        
        # Сортируем по убыванию оценки и берем топ-N
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_movies = movie_scores[:n]
        
        # Получаем информацию о фильмах из базы данных
        recommendations = []
        for movie_index, predicted_rating in top_movies:
            movie_id = self.index_to_movie[movie_index]
            try:
                movie = Movie.objects.get(id=movie_id)
                recommendation = {
                    'movie_id': movie_id,
                    'title': movie.movie_name,
                    'predicted_rating': round(predicted_rating, 2)
                }
                recommendations.append(recommendation)
                print(
                    f"{recommendation['movie_id']:2d}. {recommendation['title']:40} | Оценка: {recommendation['predicted_rating']:.0f}"
                )
            except Movie.DoesNotExist:
                print(f"Фильм с ID {movie_id} не найден в базе данных")
                continue
        
        return recommendations

def main():
    user = User.objects.first()
    if user:
        recommender = MovieRecommender()
        recommender.train()
        user_pred = recommender.predict_user_based_naive(top=5)
        print('User similarity matrix:')
        print(recommender.user_similarity)
        if isinstance(user_pred, bool):
            return
        print('UserToUser RMSE:', recommender.rmse(user_pred, recommender.test_data_matrix))
        precision, recall = recommender.precision_recall_at_k(user_pred, recommender.test_data_matrix)
        print(f'UserToUser Precision = {precision:.2f}. Recall = {recall:.2f}')
        print(f'UserToUser F1-score: {recommender.f1_score(user_pred, recommender.test_data_matrix):.2f}')
        user_pred = recommender.predict_user_based_k_fract_mean(top=5)
        print('User similarity matrix:')
        print(recommender.user_similarity)
        if isinstance(user_pred, bool):
            return
        print('UserToUser RMSE:', recommender.rmse(user_pred, recommender.test_data_matrix))
        precision, recall = recommender.precision_recall_at_k(user_pred, recommender.test_data_matrix)
        print(f'UserToUser Precision = {precision:.2f}. Recall = {recall:.2f}')
        print(f'UserToUser F1-score: {recommender.f1_score(user_pred, recommender.test_data_matrix):.2f}')
        user_id = User.objects.get(username='user_189_2').id
        favorites = Favorite.objects.filter(user_id=user_id).select_related('movie')
        print('Избранные фильмы пользователя')
        for f in favorites:
            print(f"{f.movie.id}. {f.movie.movie_name}")
        ratings = MovieRating.objects.filter(user_id=user_id).select_related('movie').order_by('-user_rating')
        print('Оценки пользователя')
        for r in ratings:
            print(f"{r.movie.id}. {r.movie.movie_name} | Оценка: {r.user_rating}")
        print('Рекоммендации пользователя')
        recommender.recommend_movies(user_id, user_pred, n=10, include_ratings=True)

if __name__ == "__main__":
    main()