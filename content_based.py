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
            return None, None, None
        #print("Оценки пользователя:\n", user_ratings, sep="")
        #print()
        #print("Избранное пользователя:\n", favorites, sep="")
        #print()
        # получаем все жанры и всех режиссеров, стран из избранных фильмов и взвешиваем их
        genre_weights = Counter()
        director_weights = Counter()
        country_weights = Counter()
        total_weight = 0

        # обрабатываем оценки (1-10)
        for rating in user_ratings:
            weight = rating.user_rating  # Вес = оценка
            movie = rating.movie
            
            if movie.genre:
                genres = [g.name.lower() for g in movie.genre.all()]
                for genre in genres:
                    genre_weights[genre] += weight
            
            if movie.director:
                directors = [d.name.lower() for d in movie.director.all()]
                for director in directors:
                    director_weights[director] += weight
            
            if movie.country:
                countries = [c.name.lower() for c in movie.country.all()]
                for country in countries:
                    country_weights[country] += weight
            
            total_weight += weight

        # обрабатываем избранное (вес = средняя оценка пользователя или 5)
        weight = user_ratings.aggregate(Avg('user_rating'))['user_rating__avg'] or 5
        for fav in Favorite.objects.filter(user=self.user).select_related('movie'):
            movie = fav.movie
            
            if movie.genre:
                genres = [g.name.lower() for g in movie.genre.all()]
                for genre in genres:
                    genre_weights[genre] += weight
            
            if movie.director:
                directors = [d.name.lower() for d in movie.director.all()]
                for director in directors:
                    director_weights[director] += weight

            if movie.country:
                countries = [c.name.lower() for c in movie.country.all()]
                for country in countries:
                    country_weights[country] += weight
            
            total_weight += weight
        
        # нормализация весов жанров
        genre_prefs = {}
        for genre, weight in genre_weights.items():
            genre_prefs[genre] = weight / total_weight
    
        # нормализация весов режиссёров
        director_prefs = {}
        for director, weight in director_weights.items():
            director_prefs[director] = weight / total_weight

        # нормализация весов стран
        country_prefs = {}
        for country, weight in country_weights.items():
            country_prefs[country] = weight / total_weight

        return genre_prefs, director_prefs, country_prefs
    
    # возвращает n число рекомендованных фильмов
    def recommend_movies(self, n=5):
        cache_key = f"user_recs_{self.user.id}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        genre_prefs, director_prefs, country_prefs = self.get_user_preferences()
        #print("Веса жанров:\n", genre_prefs, sep="")
        #print()
        #print("Веса режиссеров:\n", director_prefs, sep="")
        #print()
        #print("Веса стран:\n", country_prefs, sep="")
        #print()
        if genre_prefs==None or director_prefs==None or country_prefs==None:
            return []
        
        # получаем все фильмы, которые пользователь еще не добавлял в избранное
        excluded_ids = Favorite.objects.filter(user=self.user).values_list('movie_id', flat=True)
        all_movies = Movie.objects.exclude(id__in=excluded_ids)
        
        # создаем список для хранения фильмов. score - показатель рекомендации: чем выше, тем более подходящий пользователю
        scored_movies = []
        
        for movie in all_movies:
            score = 0
            # оценка по жанрам
            if genre_prefs:
                movie_genres = [g.name.lower() for g in movie.genre.all()]
                # вычисляем оценку на основе совпадения жанров
                for genre, pref in genre_prefs.items():
                    if genre in movie_genres:
                        score += pref * 0.6
            # оценка по режиссерам
            if director_prefs:
                movie_directors = [d.name.lower() for d in movie.director.all()]
                # вычисляем оценку на основе совпадения режиссеров
                for director, pref in director_prefs.items():
                    if director in movie_directors:
                        score += pref * 0.4
            # оценка по странам
            if country_prefs:
                movie_countries = [c.name.lower() for c in movie.country.all()]
                # вычисляем оценку на основе совпадения стран
                for country, pref in country_prefs.items():
                    if country in movie_countries:
                        score += pref * 0.3
            
            # учитываем рейтинг фильма
            if movie.rating:
                score *= (movie.rating / 10)  # нормализуем рейтинг
            
            if score > 0:
                scored_movies.append((movie, score))
        
        # сортируем по убыванию оценки
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        #print("Фильмы с оценками score:")
        #for movie in scored_movies:
        #    print(movie)
        # возвращаем топ-N фильмов
        recommendations = [movie for movie, score in scored_movies[:n]]
        cache.set(cache_key, recommendations, timeout=3)
        return recommendations