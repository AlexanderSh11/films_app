from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import os
import random
import django
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

# класс Кластера
class Cluster:

    def __init__(self, name: str, center: list, std: float):
        self.name = name
        self.center = center # центроид класса в виде списка чисел
        self.std = std # стандартное отклонение

    def get_params(self):
        return {'center': self.center, 'std': self.std}

# класс Генератора пользователей
class UsersGenerator:

    def __init__(self, clusters: list, features: list, n_users=100):
        self.n_users = n_users # количество генерируемых пользователей
        self.clusters = clusters # список кластеров
        self.features = features # список названий признаков центроидов
        self.centers = [cluster.center for cluster in self.clusters] # список центроидов кластеров
        self.std_list = [cluster.std for cluster in self.clusters] # список стандартных отклонений кластеров

    # генерация синтетических данных (пользователей)
    def generate_data(self):
        X, y = make_blobs(
            n_samples=self.n_users, 
            n_features=len(self.features), 
            centers=self.centers, 
            cluster_std=self.std_list, 
            random_state=42
        )
        # преобразование данных в Датафрейм, сохранение в атрибут data
        self.data = pd.DataFrame(X, columns=self.features)
        # вставка в Датафрейм колонки с номерами кластеров
        self.data.insert(loc=len(self.data.columns), column='Real_Cluster', value=y)
        return self.data
    
    def get_data(self):
        return self.data
    
    # обучение алгоритма k-means
    def fit_kmeans(self):
        model_kmeans = KMeans(n_clusters=len(self.centers), random_state=42)
        predicted_clusters = model_kmeans.fit_predict(self.data[self.features])
        self.data['Predicted_Cluster'] = predicted_clusters
        # сопоставление кластеров по номерам
        # создаем матрицу ошибок
        cm = confusion_matrix(self.data["Real_Cluster"], self.data["Predicted_Cluster"])
        # задаем соответствие реального и предсказанного кластера по уменьшению ошибок
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = dict(zip(col_ind, row_ind))
        # переопределяем предсказанные метки
        self.data["Predicted_Cluster"] = self.data["Predicted_Cluster"].map(mapping)
        # проверяем совпадения
        match_rate = (self.data["Real_Cluster"] == self.data["Predicted_Cluster"]).mean()
        print(f"Совпадение кластеров: {match_rate:.2%}")
        centroids = model_kmeans.cluster_centers_
        return centroids
    
    # функция получения данных из CSV файла
    def get_from_csv(self, filename):
        self.data = pd.read_csv(filename, encoding="utf-8")
        return self.data

    # функция сохранения данных в CSV файл
    def save_csv(self, filename):
        self.data.to_csv(filename, index=False, encoding="utf-8")

    # Генерация избранных фильмов и оценок для пользователя на основе профиля
    def generate_user_activity(self, user_row, movies, n_favorites=None, n_rated=None):
        """
        user_row: строка DataFrame пользователя
        movies: список объектов Movie
        n_favorites: количество избранных
        n_rated: количество оцененных фильмов
        """
        # если не задано количество избранных фильмов пользователя
        if n_favorites is None:
            # выбираем минимальное среди количества фильмов и значения favorites_count
            n_favorites = int(min(len(movies), user_row.get('favorites_count', 10)))
        # аналогично с количеством оценок пользователя
        if n_rated is None:
            n_rated = int(min(len(movies), user_row.get('count_ratings', 10)))

        # списки для кортежей (фильм, score)
        favorites_candidates = []
        rated_candidates = []

        # цикл по всем фильмам
        for movie in movies:
            # получаем список жанров, режиссеров, стран фильма
            movie_genres = [g.name.lower() for g in movie.genre.all()]
            movie_directors = [d.name.lower() for d in movie.director.all()]
            movie_countries = [c.name.lower() for c in movie.country.all()]
            # получаем год фильма, его продолжительность в минутах, оценку и возврастной рейтинг
            movie_year = movie.year
            movie_runtime = int(movie.runtime.split()[0])
            movie_rating = movie.rating
            movie_age_rating = getattr(movie, 'age_rating', None)

            score = 0
            # добавляем score при совпадении жанра, пользователя и страны с характеристикой пользователя
            if str(user_row['favorite_genre']).lower() in movie_genres:
                score += 7
            if str(user_row['high_rated_genre']).lower() in movie_genres:
                score += 5
            if str(user_row['favorite_director']).lower() in movie_directors:
                score += 10
            if str(user_row['favorite_country']).lower() in movie_countries:
                score += 3

            # оценка по десятилетиям
            if movie_year >= 1970:
                decade = (movie_year // 10) * 10
                decade_key = f"decade_count_{decade % 100:02d}"
                score += user_row.get(decade_key, 0) * 0.7

            # по времени и рейтингу (чем ближе к значению, тем выше score)
            runtime_diff = abs(user_row.get('avg_runtime', 100) - movie_runtime)
            score += max(0, 5 - runtime_diff/15)

            rating_diff = abs(user_row.get('avg_rating', 5) - movie_rating)
            score += max(0, 5 - rating_diff)

            if movie_age_rating is not None:
                age_diff = abs(user_row.get('avg_age_rating', 12) - movie_age_rating)
                score += max(0, 5 - age_diff / 3)

            if score > 0:
                favorites_candidates.append((movie, score))
            rated_candidates.append((movie, score))

        # сортировка по убыванию score
        # добавление n_favorites фильмов в список избранных пользователя
        favorites_candidates.sort(key=lambda x: x[1], reverse=True)
        favorites = [m for m, _ in favorites_candidates[:n_favorites]]

        # выбираем случайные фильмы для оценки
        if len(movies) > n_rated:
            rated = random.sample(rated_candidates, n_rated)
        else:
            rated = rated_candidates

        # нормализация оценок от 1 до 10
        scores = np.array([s for _, s in rated])
        if len(scores) == 0:
            ratings = {}
        else:
            min_s, max_s = scores.min(), scores.max()
            if max_s == min_s:
                ratings = {m.id: random.randint(4, 7) for m, _ in rated}
            else:
                ratings = {m.id: int(round(1 + (s - min_s) / (max_s - min_s) * 9)) for m, s in rated}

        return {"favorites_ids": [m.id for m in favorites], "ratings": ratings}
    
    # вывод информации о пользователе
    @staticmethod
    def print_user_info(user_info, n=5):
        """
        user_info: один элемент из user_data
        n: сколько первых избранных фильмов и оценок выводить
        """
        print(f"Информация о пользователе {user_info['user_id']}")
        print(f"Кластер: {user_info['cluster']}")
        print("Характеристики профиля:")
        for key, val in user_info["profile"].items():
            print(f"  {key}: {val}")

        print(f"\nИзбранных фильмов: {len(user_info['favorites'])}")
        print(f"Первые {n} избранных:")
        for mid in user_info["favorites"][:n]:
            print(f"  Фильм ID {mid}")

        print(f"\nОценённых фильмов: {len(user_info['ratings'])}")
        print(f"Первые {n} оценок:")
        for mid, r in list(user_info["ratings"].items())[:n]:
            print(f"  Фильм ID {mid}: оценка {r}")
            
    def save_data_to_db(self, user_data):
        from django.contrib.auth.models import User
        from django.contrib.auth.hashers import make_password
        from django.db import transaction
        from app.models import Favorite, MovieRating
        # для обеспечения целостности данных
        with transaction.atomic():
            password = make_password('password_user')
            for user_info in user_data:
                try:
                    # создаем пользователя
                    username = f"user_{user_info['user_id']}"
                    email = f"{username}@example.com"
                    
                    user, created = User.objects.get_or_create(
                        username=username,
                        defaults={
                            'email': email,
                            'password': password
                        }
                    )
                    if not created:
                        print(f"Пользователь {username} уже существует, обновляем данные")
                    
                    # сохраняем избранные фильмы
                    favorites_created = 0
                    for movie_id in user_info['favorites']:
                        try:
                            favorite, fav_created = Favorite.objects.get_or_create(
                                user=user,
                                movie_id=movie_id
                            )
                            if fav_created:
                                favorites_created += 1
                        except Exception as e:
                            print(f"Ошибка при создании избранного для пользователя {username}, фильм {movie_id}: {e}")
                    
                    # сохраняем оценки фильмов
                    ratings_created = 0
                    for movie_id, rating_value in user_info['ratings'].items():
                        try:
                            rating, rating_created = MovieRating.objects.get_or_create(
                                user=user,
                                movie_id=movie_id,
                                defaults={'user_rating': rating_value}
                            )
                            if rating_created:
                                ratings_created += 1
                        except Exception as e:
                            print(f"Ошибка при создании оценки для пользователя {username}, фильм {movie_id}: {e}")
                    
                except Exception as e:
                    print(f"Ошибка при обработке пользователя {user_info['user_id']}: {e}")
                    continue
    

# класс Визуализатор
class Visualizer:
    # вывод диаграммы реальных кластеров пользователей
    @staticmethod
    def plot_2d(df, x, y, cluster_col, colors=None, title=""):
        plt.figure(figsize=(8,6))
        if colors is not None:
            plt.scatter(df[x], df[y], c=colors, s=100)
        else:
            sns.scatterplot(data=df, x=x, y=y, hue=cluster_col, palette="tab10", s=100)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.savefig("users_generated_2d.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # вывод диаграммы реальных и предсказанных кластеров пользователей
    @staticmethod
    def compare_real_pred(df, x, y, real_col='Real_Cluster', pred_col='Predicted_Cluster',
                          colors=None, centers=None):
        fig, axes = plt.subplots(1, 2, figsize=(14,6))
        if colors is not None:
            axes[0].scatter(df[x], df[y], c=colors, s=100)
            axes[1].scatter(df[x], df[y], c=colors, s=100)
        else:
            sns.scatterplot(data=df, x=x, y=y, hue=real_col, palette="tab10", s=100, ax=axes[0])
            sns.scatterplot(data=df, x=x, y=y, hue=pred_col, palette="tab10", s=100, ax=axes[1])

        if centers is not None:
            feature_names = list(df.columns)
            if feature_names is not None:
                # находим индексы нужных признаков
                idx_x = feature_names.index(x)
                idx_y = feature_names.index(y)
                centers_xy = centers[:, [idx_x, idx_y]]
            else:
                # если список признаков не передан — предполагаем, что первые два измерения нужны
                centers_xy = centers[:, :2]

            axes[1].scatter(centers_xy[:, 0], centers_xy[:, 1],
                            s=150, color="red", marker='X', edgecolor='black', linewidth=1.2)
        axes[0].set_title("Real Clusters")
        axes[1].set_title("Predicted Clusters")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.savefig("users_generated_2d_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

    # вывод диаграммы реальных кластеров пользователей в трехмерном измерении
    @staticmethod
    def plot_3d(df, x, y, z, cluster_col, colors=None):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        if colors is not None:
            scatter = ax.scatter(df[x], df[y], df[z], c=colors, s=50)
        else:
            scatter = ax.scatter(df[x], df[y], df[z], c=df[cluster_col], cmap='tab10', s=50)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.savefig("users_generated_3d.png", dpi=300, bbox_inches='tight')
        plt.show()

# функция нормализации данных
def normalize_data(df, category_maps, numeric_cols, age_rating_col='avg_age_rating'):
    df_norm = df.copy()

    # нормализация категорий
    df_norm = decode_categories(df_norm, category_maps)

    # нормализация числовых признаков
    for col in numeric_cols:
        if col != age_rating_col:
            # округляем и убираем отрицательные значения
            df_norm[col] = df_norm[col].round().clip(lower=0).astype(int)

    # специальная нормализация для avg_age_rating
    age_values = np.array([0, 6, 12, 16, 18])
    df_norm[age_rating_col] = df_norm[age_rating_col].apply(
        lambda x: int(age_values[np.argmin(np.abs(age_values - max(0, x)))])
    )

    return df_norm


# декодирование категориальных признаков
def decode_categories(df, category_maps):
    decoded = df.copy()
    for col, mapping in category_maps.items():
        # отсортируем диапазоны по возрастанию нижней границы
        sorted_ranges = sorted(mapping.keys(), key=lambda x: x[0])
        decoded[col] = decoded[col].astype(object)
        for idx, val in decoded[col].items():
            if val < sorted_ranges[0][0]:
                # если меньше минимального диапазона — первая категория
                decoded.at[idx, col] = mapping[sorted_ranges[0]]
            elif val > sorted_ranges[-1][1]:
                # если больше максимального диапазона — последняя категория
                decoded.at[idx, col] = mapping[sorted_ranges[-1]]
            else:
                # ищем диапазон, куда входит число
                for r, name in mapping.items():
                    if r[0] <= val <= r[1]:
                        decoded.at[idx, col] = name
                        break
    return decoded


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myapp.settings")
    django.setup()

    from app.models import Movie

    category_maps = {
        "favorite_genre": {
            (0, 10): "Фантастика",
            (10, 20): "Боевик",
            (20, 30): "Приключения",
            (30, 40): "Драма",
            (40, 50): "Комедия",
            (50, 60): "Аниме",
            (60, 70): "Криминал",
            (70, 80): "Семейный",
            (80, 90): "Ужасы"
        },
        "high_rated_genre": {
            (0, 20): "Фантастика",
            (20, 40): "Боевик",
            (40, 60): "Триллер",
            (60, 80): "Драма",
            (80, 100): "Комедия"
        },
        "favorite_director": {
            (0, 10): "Кристофер Нолан",
            (10, 20): "Джеймс Кэмерон",
            (20, 30): "Питер Джексон",
            (30, 40): "Хаяо Миядзаки",
            (40, 50): "Леонид Гайдай",
            (50, 60): "Квентин Тарантино",
            (60, 70): "Гай Ричи",
            (70, 80): "Роберт Земекис",
            (80, 90): "Стивен Роач"
        },
        "favorite_country": {
            (0, 30): "США",
            (30, 50): "Великобритания",
            (50, 60): "Канада",
            (60, 80): "Россия",
            (80, 90): "СССР",
            (90, 100): "Япония"
        }
    }
    numeric_limits = {
        'decade_count_70': (0, 10),
        'decade_count_80': (0, 10),
        'decade_count_90': (0, 10),
        'decade_count_00': (0, 10),
        'decade_count_10': (0, 10),
        'decade_count_20': (0, 10),
        'count_ratings': (5, 50),
        'max_rating_count': (1, 40),
        'favorites_count': (5, 40),
        'avg_runtime': (50, 150),
        'avg_age_rating': (0, 18)
    }
    clusters = [
        Cluster(
            name="Любители супергероев",
            center=[5, 10, 5, 10, 3, 4, 5, 6, 2, 4, 35, 25, 20, 110, 16],
            std=5
        ),
        Cluster(
            name="Любители отечественного кино",
            center=[45, 35, 45, 70, 4, 5, 6, 5, 1, 0, 25, 20, 15, 90, 12],
            std=5
        ),
        Cluster(
            name="Любители аниме",
            center=[55, 50, 55, 95, 2, 3, 4, 5, 5, 5, 40, 30, 25, 80, 6],
            std=5
        ),
        Cluster(
            name="Любители криминала",
            center=[65, 45, 65, 40, 3, 5, 4, 6, 1, 1, 30, 25, 20, 110, 16],
            std=5
        ),
        Cluster(
            name="Любители семейных фильмов",
            center=[75, 70, 70, 20, 1, 2, 3, 4, 0, 0, 20, 15, 25, 100, 6],
            std=5
        ),
        Cluster(
            name="Любители ужасов",
            center=[85, 75, 80, 20, 2, 3, 4, 5, 1, 1, 30, 20, 20, 90, 18],
            std=5
        )
    ]
    features = [
        'favorite_genre', 'high_rated_genre', 'favorite_director', 'favorite_country',
        'decade_count_70', 'decade_count_80', 'decade_count_90', 'decade_count_00',
        'decade_count_10', 'decade_count_20', 'count_ratings', 'max_rating_count', 'favorites_count',
        'avg_runtime', 'avg_age_rating'
    ]
    N_users = 1000
    generator = UsersGenerator(clusters=clusters, n_users=1000, features=features)
    data = generator.generate_data()
    print("Сгенерированы пользователи.")
    centroids = generator.fit_kmeans()

    # подготовим цвета по кластерам
    unique_clusters = sorted(data['Real_Cluster'].unique())
    palette = sns.color_palette('Set2', n_colors=len(unique_clusters))
    color_map = {c: palette[i] for i, c in enumerate(unique_clusters)}
    colors = [color_map[c] for c in data['Real_Cluster']]

    # Визуализируем реальные кластеры
    Visualizer.compare_real_pred(
        data, 
        x='avg_runtime',
        y='favorite_genre', 
        real_col='Real_Cluster',
        pred_col='Predicted_Cluster',
        colors=colors,
        centers=centroids
    )

    # Визуализируем предсказанные кластеры
    Visualizer.plot_3d(
        data, 
        x='avg_runtime', 
        y='favorite_genre', 
        z='avg_age_rating', 
        cluster_col='Predicted_Cluster',
        colors=colors
    )
    print("Визуализированы кластеры.")
    movies = list(Movie.objects.all()[:200])

    user_data = []
    numeric_cols = [
        'decade_count_70', 'decade_count_80', 'decade_count_90', 'decade_count_00', 'decade_count_10', 'decade_count_20',
        'count_ratings', 'max_rating_count', 'favorites_count', 'avg_runtime', 'avg_age_rating'
    ]
    data = normalize_data(data, category_maps, numeric_cols)
    for idx, row in data.iterrows():
        activity = generator.generate_user_activity(row, movies)
        user_data.append({
            "user_id": idx + 1,
            "cluster": row["Real_Cluster"],
            "profile": row.to_dict(),
            "favorites": activity["favorites_ids"],
            "ratings": activity["ratings"]
        })
        if idx % 10 == 0:
            print(f"Сгенерировано оценок и избранных фильмов для {idx+1} пользователей.")
        if idx >= N_users:
            break

    generator.print_user_info(user_data[-1])
    generator.save_data_to_db(user_data)

if __name__ == "__main__":
    main()