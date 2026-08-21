from app.models import Movie, Favorite, MovieRating
from django.contrib.auth.models import User
from content_based import MovieRecommender as ContentBasedRecommender
from collaborative_filtering import MovieRecommender as CollaborativeRecommender

import os
import django
from django.core.cache import cache

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myapp.settings")
django.setup()


class HybridRecommender:
    def __init__(self):
        self.collaborative = CollaborativeRecommender()
        self.is_trained = False

    def train(self, verbose=False):
        self.collaborative.train(verbose=verbose)
        self.is_trained = True
        return True

    def recommend_hybrid(
        self, user_id, n=10, cf_weight=0.5, cb_weight=0.5, verbose=False
    ):
        """
        Гибридные рекомендации
        cf_weight - вес коллаборативной фильтрации (0-1)
        cb_weight - вес контентной фильтрации (0-1)
        cf_weight + cb_weight = 1
        """
        cache_key = f"hybrid_recs_{user_id}_{n}_{cf_weight}_{cb_weight}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Получаем пользователя
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return []

        # Нормализуем веса
        total_weight = cf_weight + cb_weight
        cf_weight_norm = cf_weight / total_weight
        cb_weight_norm = cb_weight / total_weight

        # Коллаборативные рекомендации
        cf_movies_with_scores = []
        if cf_weight > 0 and self.is_trained:
            if user_id in self.collaborative.user_to_index:
                pred = self.collaborative.predict_user_based_k_fract_mean(top=100)
                if not isinstance(pred, bool):
                    cf_movies = self.collaborative.recommend_movies(
                        user_id, pred, n=n * 3
                    )
                    # Присваиваем веса на основе позиции
                    for i, movie in enumerate(cf_movies):
                        score = 1.0 - (i / (n * 3)) * 0.5  # от 1.0 до 0.5
                        cf_movies_with_scores.append((movie, score))

        # Контентные рекомендации
        cb_movies_with_scores = []
        if cb_weight > 0:
            cb = ContentBasedRecommender(user)
            cb_movies = cb.recommend_movies(n=n * 3)
            for i, movie in enumerate(cb_movies):
                score = 1.0 - (i / (n * 3)) * 0.5  # от 1.0 до 0.5
                cb_movies_with_scores.append((movie, score))

        # Объединяем с весами
        movie_scores = {}

        for movie, score in cf_movies_with_scores:
            movie_scores[movie.id] = (
                movie_scores.get(movie.id, 0) + score * cf_weight_norm
            )

        for movie, score in cb_movies_with_scores:
            movie_scores[movie.id] = (
                movie_scores.get(movie.id, 0) + score * cb_weight_norm
            )

        # Исключаем уже просмотренные
        excluded_ids = set(
            list(Favorite.objects.filter(user=user).values_list("movie_id", flat=True))
            + list(
                MovieRating.objects.filter(user=user).values_list("movie_id", flat=True)
            )
        )

        # Сортируем и получаем топ-N
        sorted_movies = sorted(
            [
                (movie_id, score)
                for movie_id, score in movie_scores.items()
                if movie_id not in excluded_ids
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        recommendations = []
        for movie_id, score in sorted_movies[:n]:
            try:
                movie = Movie.objects.get(id=movie_id)
                recommendations.append(movie)
                if verbose:
                    print(
                        f"{movie.id:2d}. {movie.movie_name:40} | Гибридный score: {score:.3f}"
                    )
            except Movie.DoesNotExist:
                continue

        cache.set(cache_key, recommendations, timeout=60 * 5)
        return recommendations

    def recommend_adaptive(self, user_id, n=10, verbose=False):
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return []

        # Считаем активность пользователя
        n_ratings = MovieRating.objects.filter(user=user).count()
        n_favorites = Favorite.objects.filter(user=user).count()
        total_activity = n_ratings + n_favorites

        if total_activity == 0:
            # Новый пользователь - только контентная (холодный старт)
            cf_weight, cb_weight = 0.0, 1.0
        elif total_activity < 5:
            # Мало активности - больше контентной
            cf_weight, cb_weight = 0.3, 0.7
        elif total_activity < 15:
            # Средняя активность - баланс
            cf_weight, cb_weight = 0.5, 0.5
        else:
            # Много активности - больше коллаборативной
            cf_weight, cb_weight = 0.7, 0.3

        if verbose:
            print(f"Активность пользователя: {total_activity}")
            print(f"Веса: CF={cf_weight:.2f}, CB={cb_weight:.2f}")

        return self.recommend_hybrid(user_id, n, cf_weight, cb_weight, verbose)


def main():
    user = User.objects.get(username="admin")
    user_id = user.id

    print(f"Информация о пользователе: {user.username} (id={user_id})")

    favorites = Favorite.objects.filter(user=user).select_related("movie")
    print(f"Избранные фильмы ({favorites.count()}):")
    for f in favorites:
        print(f"  {f.movie.id}. {f.movie.movie_name}")

    ratings = (
        MovieRating.objects.filter(user=user)
        .select_related("movie")
        .order_by("-user_rating")
    )
    print(f"Оценки ({ratings.count()}):")
    for r in ratings:
        print(f"  {r.movie.id}. {r.movie.movie_name} | Оценка: {r.user_rating}")

    # Создаем и обучаем гибридную модель
    hybrid = HybridRecommender()
    hybrid.train(verbose=True)

    # Только коллаборативная фильтрация
    print("1. Коллаборативная фильтрация:")
    if user_id in hybrid.collaborative.user_to_index:
        pred = hybrid.collaborative.predict_user_based_k_fract_mean(top=100)
        if not isinstance(pred, bool):
            cf_recs = hybrid.collaborative.recommend_movies(
                user_id, pred, n=5, verbose=True
            )

    # Только контентная фильтрация
    print("2. Контентная фильтрация:")
    cb = ContentBasedRecommender(user)
    cb_recs = cb.recommend_movies(n=5)
    for i, movie in enumerate(cb_recs, 1):
        print(f"  {i:2d}. {movie.movie_name}")

    # Гибридные рекомендации (сбалансированные)
    print("3. Гибридные рекомендации (CF=0.5, CB=0.5):")
    hybrid_recs_balanced = hybrid.recommend_hybrid(
        user_id, n=5, cf_weight=0.5, cb_weight=0.5, verbose=True
    )

    # Адаптивные гибридные рекомендации
    print("4. Адаптивные гибридные рекомендации:")
    hybrid_recs_adaptive = hybrid.recommend_adaptive(user_id, n=5, verbose=True)

    # Анализ пересечений
    cf_ids = {m.id for m in cf_recs} if "cf_recs" in locals() else set()
    cb_ids = {m.id for m in cb_recs}
    hybrid_ids = {m.id for m in hybrid_recs_adaptive}

    print(f"CF И CB: {len(cf_ids & cb_ids)} фильмов")
    print(f"CF И Гибрид: {len(cf_ids & hybrid_ids)} фильмов")
    print(f"CB И Гибрид: {len(cb_ids & hybrid_ids)} фильмов")
    print(f"Уникальные для гибрида: {len(hybrid_ids - cf_ids - cb_ids)} фильмов")


if __name__ == "__main__":
    main()
