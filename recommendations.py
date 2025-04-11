from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
from django.core.cache import cache
from app.models import Movie, Favorite, MovieRating

class MovieRecommender:
    def __init__(self, user):
        self.user = user
    
    # получить предпочтения пользователя в жанрах и режиссерах
    def get_user_preferences(self):

        favorites_count = Favorite.objects.filter(user=self.user).count()
        if favorites_count < 5:  # минимум 5 фильмов для анализа
            return None, None

        favorites = Favorite.objects.filter(user=self.user).select_related('movie')
        
        # получаем все жанры из избранных фильмов
        all_genres = []
        all_directors = []
        for fav in favorites:
            if fav.movie.genre:
                genres = [g.strip().lower() for g in fav.movie.genre.split(',')]
                all_genres.extend(genres)

            if fav.movie.director:
                directors = [d.strip().lower() for d in fav.movie.director.split(',')]
                all_directors.extend(directors)
        
        # нормализация жанров
        genre_prefs = {}
        if all_genres:
            # подсчитываем частоту жанров
            genre_counter = Counter(all_genres)
            total = len(all_genres)
            # преобразуем в словарь, сортируем по убыванию популярности
            for genre, count in genre_counter.most_common():
                genre_prefs[genre] = count/total
    
        # нормализация режиссёров
        director_prefs = {}
        if all_directors:
            # подсчитываем частоту режиссеров
            director_counter = Counter(all_directors)
            total = len(all_directors)
            # преобразуем в словарь, сортируем по убыванию популярности
            for director, count in director_counter.most_common():
                director_prefs[director] = count/total

        return genre_prefs, director_prefs
    
    # возвращает n число рекомендованных фильмов
    def recommend_movies(self, n=5):
        cache_key = f"user_recs_{self.user.id}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        genre_prefs, director_prefs = self.get_user_preferences()
        
        if genre_prefs==None or director_prefs==None:
            return []
        
        # получаем все фильмы, которые пользователь еще не добавлял в избранное
        excluded_ids = Favorite.objects.filter(user=self.user).values_list('movie_id', flat=True)
        all_movies = Movie.objects.exclude(id__in=excluded_ids)
        
        # создаем список для хранения фильмов. score - показатель рекоммендации: чем выше, тем более подходящий пользователю
        scored_movies = []
        
        for movie in all_movies:
            score = 0
            # оценка по жанрам
            if genre_prefs:
                movie_genres = [g.strip().lower() for g in movie.genre.split(',')]
                # вычисляем оценку на основе совпадения жанров
                for genre, pref in genre_prefs.items():
                    if genre in movie_genres:
                        score += pref * 0.7
            # оценка по режиссерам
            if director_prefs:
                movie_directors = [d.strip().lower() for d in movie.director.split(',')]
                # вычисляем оценку на основе совпадения режиссеров
                for director, pref in director_prefs.items():
                    if director in movie_directors:
                        score += pref * 0.3
            
            # учитываем рейтинг фильма
            if movie.rating:
                score *= (movie.rating / 10)  # нормализуем рейтинг
            
            if score > 0:
                scored_movies.append((movie, score))
        
        # сортируем по убыванию оценки
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        # возвращаем топ-N фильмов
        recommendations = [movie for movie, score in scored_movies[:n]]
        cache.set(cache_key, recommendations, timeout=300)
        return recommendations