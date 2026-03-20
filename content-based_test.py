import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myapp.settings')

import django
django.setup()

from django.contrib.auth.models import User
from app.models import Movie, Favorite, MovieRating
from content_based import MovieRecommender


def create_test_user(username, password='testpass123'):
    """Создание тестового пользователя"""
    user, created = User.objects.get_or_create(
        username=username,
        defaults={'email': f'{username}@test.com'}
    )
    if created:
        user.set_password(password)
        user.save()
        print(f"Создан пользователь: {username}")
    else:
        print(f"Пользователь {username} уже существует, очищаем данные...")
        MovieRating.objects.filter(user=user).delete()
        Favorite.objects.filter(user=user).delete()
    return user


def add_to_favorites(user, movie_titles):
    """Добавление фильмов в избранное"""
    added = []
    for title in movie_titles:
        try:
            movie = Movie.objects.get(movie_name__iexact=title)
            _, created = Favorite.objects.get_or_create(user=user, movie=movie)
            if created:
                added.append(title)
        except Movie.DoesNotExist:
            print(f"Фильм не найден: {title}")
    print(f"Добавлено в избранное: {len(added)} из {len(movie_titles)}")
    return added


def add_ratings(user, ratings_dict):
    """Добавление оценок фильмам"""
    added = []
    for title, score in ratings_dict.items():
        try:
            movie = Movie.objects.get(movie_name__iexact=title)
            MovieRating.objects.update_or_create(
                user=user,
                movie=movie,
                defaults={'user_rating': score}
            )
            added.append((title, score))
        except Movie.DoesNotExist:
            print(f"Фильм не найден: {title}")
    print(f"Добавлено оценок: {len(added)} из {len(ratings_dict)}")
    return added


def print_user_stats(user):
    """Вывод статистики пользователя"""
    favorites = Favorite.objects.filter(user=user)
    ratings = MovieRating.objects.filter(user=user)
    print(f"Статистика пользователя {user.username}:")
    print(f"- Избранных фильмов: {favorites.count()}")
    print(f"- Оценки: {ratings.count()}")


def print_user_preferences(recommender):
    """Вывод весов предпочтений пользователя"""
    genre_prefs, director_prefs, country_prefs = recommender.get_user_preferences()
    if genre_prefs is None:
        print("Недостаточно данных для формирования профиля (менее 10 взаимодействий)")
        return

    print(f"Пользователь: {recommender.user.username}:")
    
    if genre_prefs:
        print("Топ-5 жанров:")
        sorted_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        for genre, weight in sorted_genres:
            print(f"- {genre}: {weight:.4f}")
    
    if director_prefs:
        print("Топ-5 режиссеров:")
        sorted_dirs = sorted(director_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        for director, weight in sorted_dirs:
            print(f"- {director}: {weight:.4f}")
    
    if country_prefs:
        print("Топ-5 стран:")
        sorted_countries = sorted(country_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        for country, weight in sorted_countries:
            print(f"- {country}: {weight:.4f}")


def print_recommendations(recommender, n=10):
    """Вывод рекомендаций для пользователя"""
    recommendations = recommender.recommend_movies(n=n)
    if not recommendations:
        print("Нет рекомендаций (недостаточно данных)")
        return []

    print(f"Рекомендации для {recommender.user.username} (топ-{len(recommendations)}):")
    for i, movie in enumerate(recommendations, 1):
        genres_str = ', '.join([g.name for g in movie.genre.all()][:3])
        directors_str = ', '.join([d.name for d in movie.director.all()][:3])
        if len(movie.genre.all()) > 3:
            genres_str += "..."
        if len(movie.director.all()) > 3:
            directors_str += "..."
        print(f"{i:2}. {movie.movie_name:<40} | {directors_str} | {genres_str}")
    return recommendations


def run_test_scenario(scenario_name, user_id, favorites=None, ratings=None):
    """Запуск одного сценария тестирования"""
    print(f"Сценарий {scenario_name}")

    user = create_test_user(user_id)

    if favorites:
        add_to_favorites(user, favorites)
    if ratings:
        add_ratings(user, ratings)

    print_user_stats(user)

    recommender = MovieRecommender(user)
    
    print_user_preferences(recommender)
    
    print_recommendations(recommender, n=10)
    
    return

# Сценарий 1: недостаточная история
SCENARIO_1_FAVORITES = [
    "Побег из Шоушенка",
    "Зеленая миля",
    "Список Шиндлера"
]

# Сценарий 2: только избранное
SCENARIO_2_FAVORITES = [
    "Интерстеллар",
    "Начало",
    "Темный рыцарь",
    "Матрица",
    "Аватар",
    "Стражи Галактики",
    "Джон Уик",
    "Бойцовский клуб",
    "Безумный Макс: Дорога ярости",
    "Мстители: Финал"
]

# Сценарий 3: только оценки
SCENARIO_3_RATINGS = {
    "Семь": 10,
    "Бойцовский клуб": 10,
    "Исчезнувшая": 9,
    "Карты, деньги, два ствола": 8,
    "Большой куш": 8,
    "Рок-н-рольщик": 7,
    "Джентльмены": 9,
    "Шерлок Холмс": 8,
    "Меч короля Артура": 6,
    "Джон Уик": 9,
    "Гадкая сестра": 2
}

# Сценарий 4: смешанная история
SCENARIO_4_FAVORITES = [
    "Начало",
    "Интерстеллар",
    "Престиж",
    "Темный рыцарь",
    "Матрица"
]
SCENARIO_4_RATINGS = {
    "Криминальное чтиво": 10,
    "Бесславные ублюдки": 9,
    "Джанго освобожденный": 9,
    "Бойцовский клуб": 9,
    "Семь": 10
}

# Сценарий 5: ранжирование
SCENARIO_5_FAVORITES = [
    "Криминальное чтиво",
    "Бесславные ублюдки",
    "Джанго освобожденный",
    "Бойцовский клуб",
    "Семь"
]
SCENARIO_5_RATINGS = {
    "Криминальное чтиво": 10,
    "Бесславные ублюдки": 9,
    "Джанго освобожденный": 9,
    "Бойцовский клуб": 9,
    "Семь": 10
}

if __name__ == "__main__":
    print("Тестирование content-based рекомендательной системы")

    run_test_scenario(
        "1: Недостаточная история (<10 фильмов)",
        "user_min",
        favorites=SCENARIO_1_FAVORITES
    )

    run_test_scenario(
        "2: Только избранное (10+ фильмов)",
        "user_fav_only",
        favorites=SCENARIO_2_FAVORITES
    )

    run_test_scenario(
        "3: Только оценки",
        "user_ratings_only",
        ratings=SCENARIO_3_RATINGS
    )

    run_test_scenario(
        "4: Смешанная история (избранное + оценки)",
        "user_mixed",
        favorites=SCENARIO_4_FAVORITES,
        ratings=SCENARIO_4_RATINGS
    )

    run_test_scenario(
        "5: Проверка ранжирования (явные предпочтения)",
        "user_ranking",
        favorites=SCENARIO_5_FAVORITES,
        ratings=SCENARIO_5_RATINGS
    )
