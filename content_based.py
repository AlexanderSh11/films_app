from collections import Counter
from django.core.cache import cache
from django.db.models import Avg
from app.models import Movie, Favorite, MovieRating

class MovieRecommender:
    def __init__(self, user):
        self.user = user
    
    # получить предпочтения пользователя в жанрах и режиссерах
    def get_user_preferences(self):
        # все оценки пользователя
        user_ratings = MovieRating.objects.filter(user=self.user).select_related('movie')
        # все избранные фильмы пользователя
        favorites = Favorite.objects.filter(user=self.user).select_related('movie')
        if user_ratings.count() + favorites.count() < 10:  # минимум 10 избранных/оцененных фильмов для анализа
            return None, None
        
        # получаем все жанры и всех режиссеров из избранных фильмов и взвешиваем их
        genre_weights = Counter()
        director_weights = Counter()
        total_weight = 0

        # обрабатываем оценки (1-5)
        for rating in user_ratings:
            weight = rating.user_rating  # Вес = оценка (1-5)
            movie = rating.movie
            
            if movie.genre:
                genres = [g.strip().lower() for g in movie.genre.split(',')]
                for genre in genres:
                    genre_weights[genre] += weight
            
            if movie.director:
                directors = [d.strip().lower() for d in movie.director.split(',')]
                for director in directors:
                    director_weights[director] += weight
            
            total_weight += weight

        # обрабатываем избранное (вес = средняя оценка пользователя или 3)
        avg_rating = user_ratings.aggregate(Avg('user_rating'))['user_rating__avg'] or 3
        for fav in Favorite.objects.filter(user=self.user).select_related('movie'):
            movie = fav.movie
            
            if movie.genre:
                genres = [g.strip().lower() for g in movie.genre.split(',')]
                for genre in genres:
                    genre_weights[genre] += avg_rating
            
            if movie.director:
                directors = [d.strip().lower() for d in movie.director.split(',')]
                for director in directors:
                    director_weights[director] += avg_rating
            
            total_weight += avg_rating
        
        # нормализация весов жанров
        genre_prefs = {}
        for genre, weight in genre_weights.items():
            genre_prefs[genre] = weight / total_weight
    
        # нормализация весов режиссёров
        director_prefs = {}
        for director, weight in director_weights.items():
            director_prefs[director] = weight / total_weight

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
        cache.set(cache_key, recommendations, timeout=3)
        return recommendations