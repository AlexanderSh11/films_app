import os
import django
from django.core.cache import cache
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict

# Глобальные переменные для кэширования
_MODEL = None
_INDEX = None
_ALL_MOVIE_IDS = None
_ALL_EMBEDDINGS = None
_ALL_EMBEDDINGS_DICT = None

def load_model_and_index():
    """Загружает модель и индекс при первом вызове"""
    global _MODEL, _INDEX, _ALL_MOVIE_IDS, _ALL_EMBEDDINGS, _ALL_EMBEDDINGS_DICT

    _MODEL = cache.get('semantic_model')
    _INDEX = cache.get('semantic_index')
    _ALL_MOVIE_IDS = cache.get('semantic_ids')
    _ALL_EMBEDDINGS = cache.get('semantic_embeddings')
    _ALL_EMBEDDINGS_DICT = cache.get('semantic_embeddings_dict')

    if _MODEL is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model\\fine_tuned_rubert_tiny2')
            print(f"Загрузка дообученной модели из {model_path}")
            _MODEL = SentenceTransformer(model_path)
            print("Дообученная модель загружена")
        except Exception as e:
            print(f"Не удалось загрузить дообученную модель: {e}")
            print("Загружаем стандартную модель")
            _MODEL = SentenceTransformer('cointegrated/rubert-tiny2')
    
    index_path = 'movie_index.faiss'
    ids_path = 'movie_ids.npy'
    
    if os.path.exists(index_path) and os.path.exists(ids_path):
        _INDEX = faiss.read_index(index_path)
        _ALL_MOVIE_IDS = np.load(ids_path)
        cache.set('semantic_model', _MODEL, 60*5)
        cache.set('semantic_index', _INDEX, 60*5)
        cache.set('semantic_ids', _ALL_MOVIE_IDS, 60*5)
        print(f"Индекс загружен, содержит {len(_ALL_MOVIE_IDS)} фильмов")
    else:
        print("Индекс не найден")
        build_faiss_index(force_rebuild=True)

    if _ALL_EMBEDDINGS is None or _ALL_EMBEDDINGS_DICT is None:
        embeddings_path = 'movie_embeddings.npy'
        if os.path.exists(embeddings_path):
            _ALL_EMBEDDINGS = np.load(embeddings_path)
            # Создаем словарь {movie_id: embedding}
            _ALL_EMBEDDINGS_DICT = {
                movie_id: _ALL_EMBEDDINGS[i] 
                for i, movie_id in enumerate(_ALL_MOVIE_IDS)
            }
            cache.set('semantic_embeddings', _ALL_EMBEDDINGS, 60*5)
            cache.set('semantic_embeddings_dict', _ALL_EMBEDDINGS_DICT, 60*5)
            print(f"Эмбеддинги загружены: {_ALL_EMBEDDINGS.shape}")
        else:
            print("Эмбеддинги не найдены")
            build_embeddings_cache()

def build_embeddings_cache():

    from app.models import Movie
    
    movies = Movie.objects.prefetch_related('genre', 'director', 'country').all()
    movie_texts = []
    movie_ids = []
    
    for movie in movies:
        text = get_movie_text_representation(movie)
        movie_texts.append(text)
        movie_ids.append(movie.id)
    
    # Создаем эмбеддинги
    embeddings = _MODEL.encode(movie_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Сохраняем
    np.save('movie_embeddings.npy', embeddings)
    np.save('movie_ids_embeddings.npy', np.array(movie_ids))
    
    global _ALL_EMBEDDINGS, _ALL_EMBEDDINGS_DICT
    _ALL_EMBEDDINGS = embeddings
    _ALL_EMBEDDINGS_DICT = {mid: embeddings[i] for i, mid in enumerate(movie_ids)}
    
    print(f"Кэш эмбеддингов создан для {len(movie_ids)} фильмов")

def get_movie_text_representation(movie):
    """Преобразует объект фильма в текстовую строку для создания эмбеддингов"""
    parts = []
    
    parts.append(f"название: {movie.movie_name}")
    parts.append(f"фильм: {movie.movie_name}")
    parts.append(f"название фильма: {movie.movie_name}")

    parts.append(f"год: {movie.year}")
    decade = (movie.year // 10) * 10
    parts.append(f"{decade}-е годы")
    parts.append(f"фильмы {decade}-х")
    
    genres = list(movie.genre.all().values_list('name', flat=True))
    if genres:
        genre_text = ', '.join(genres)
        parts.append(f"жанры: {genre_text}")
        parts.append(f"жанр: {genre_text}")

        for genre in genres:
            parts.append(f"в жанре {genre}")
            parts.append(f"{genre} фильм")
            parts.append(f"{genre} кино")

            parts.append(f"{movie.year} {genre}")
            parts.append(f"{genre} {movie.year}")
            parts.append(f"фильмы {genre} {movie.year}")
            
            parts.append(f"{genre} {decade}-х")
    
    directors = list(movie.director.all().values_list('name', flat=True))
    if directors:
        director_text = ', '.join(directors)
        parts.append(f"режиссёр: {director_text}")
        parts.append(f"режиссер: {director_text}")
        for director in directors:
            parts.append(f"фильмы {director}")
            parts.append(f"режиссёр {director}")
            parts.append(f"кино {director}")

            for genre in genres:
                parts.append(f"{director} {genre}")
                parts.append(f"{genre} {director}")
                parts.append(f"фильмы {director} в жанре {genre}")
    
    countries = list(movie.country.all().values_list('name', flat=True))
    if countries:
        parts.append(f"страна: {', '.join(countries)}")
        for country in countries:
            parts.append(f"{country} кино")
            parts.append(f"фильмы {country}")

            for genre in genres:
                parts.append(f"{country} {genre}")
                parts.append(f"{genre} {country}")
                parts.append(f"{country} фильмы в жанре {genre}")
            
            parts.append(f"{country} {movie.year}")
            parts.append(f"{country} {decade}-х")
    
    if movie.overview:
        parts.append(f"описание: {movie.overview}")
    
    if movie.runtime:
        parts.append(f"длительность: {movie.runtime}")
    
    if movie.rating:
        parts.append(f"рейтинг: {movie.rating}")
        if movie.rating >= 8:
            parts.append("высокий рейтинг")
            parts.append("лучшие фильмы")
    
    if movie.meta_score:
        parts.append(f"meta_score: {movie.meta_score}")
    
    if movie.age_rating is not None:
        parts.append(f"возрастное ограничение: {movie.age_rating}+")
    
    return ". ".join(parts)

def search_movies(query, top_k=5, movie_ids=None):
    """Ищет фильмы по запросу"""
    from app.models import Movie
    # Загружаем модель и индекс (если ещё не загружены)
    load_model_and_index()
    # Создаем эмбеддинг из запроса
    query_embedding = _MODEL.encode([query], convert_to_numpy=True)
    
    # Определяем, по какому набору ID ищем
    if movie_ids is not None:
        # Поиск по подмножеству фильмов
        # Получаем эмбеддинги только для указанных ID
        embeddings = []
        valid_movie_ids = []
        
        for mid in movie_ids:
            if mid in _ALL_EMBEDDINGS_DICT:
                embeddings.append(_ALL_EMBEDDINGS_DICT[mid])
                valid_movie_ids.append(mid)
        
        if not embeddings:
            return [], []
        
        embeddings = np.array(embeddings)
        
        # Создаем временный индекс
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(embeddings)
        
        # Ищем
        distances, indices = temp_index.search(query_embedding, min(top_k, len(valid_movie_ids)))
        
        # Получаем ID результатов в порядке релевантности
        result_ids = [valid_movie_ids[i] for i in indices[0]]
        result_distances = distances[0]
        
    else:
        # Поиск по всему индексу
        distances, indices = _INDEX.search(query_embedding, top_k)
        result_ids = [_ALL_MOVIE_IDS[i] for i in indices[0]]
        result_distances = distances[0]
    
    # Получаем фильмы из БД
    movies = Movie.objects.filter(id__in=result_ids).prefetch_related('genre', 'director', 'country')
    
    # Сортируем по порядку из результатов поиска
    movie_dict = {movie.id: movie for movie in movies}
    ordered_movies = []
    ordered_distances = []
    
    for mid, dist in zip(result_ids, result_distances):
        if mid in movie_dict:
            ordered_movies.append(movie_dict[mid])
            ordered_distances.append(dist)
    
    # Применяем бустинг для названий (одинаково для обоих случаев)
    query_lower = query.lower()
    boosted_distances = list(ordered_distances.copy())
    
    for i, movie in enumerate(ordered_movies):
        title_lower = movie.movie_name.lower()
        
        # Если название полностью совпадает с запросом
        if query_lower == title_lower:
            boosted_distances[i] *= 0.3
        # Если запрос является частью названия или название часть запроса
        elif query_lower in title_lower or title_lower in query_lower:
            boosted_distances[i] *= 0.5

    # Сортируем заново с учетом бустинга
    sorted_indices = np.argsort(boosted_distances)
    final_movies = [ordered_movies[i] for i in sorted_indices]
    final_distances = [boosted_distances[i] for i in sorted_indices]

    return final_movies, final_distances

def build_faiss_index(force_rebuild=False):
    """Создает индекс FAISS для всех фильмов"""
    from app.models import Movie
    index_path = 'movie_index.faiss'
    ids_path = 'movie_ids.npy'
    
    # Проверяем существует ли уже индекс
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(ids_path):
        return
    
    # Загружаем модель cointegrated/rubert-tiny2
    model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model\\fine_tuned_rubert_tiny2')
    model = SentenceTransformer(model_path)
    
    print("Создание текстового описания для каждого фильма")
    movies = Movie.objects.prefetch_related('genre', 'director', 'country').all()
    movie_texts = []
    movie_ids = []
    for i, movie in enumerate(movies, 1):
        text = get_movie_text_representation(movie)
        movie_texts.append(text)
        movie_ids.append(movie.id)
    
    print("Создание эмбеддингов")
    embeddings = model.encode(movie_texts, convert_to_numpy=True, show_progress_bar=True)
    
    print("Создание индекса FAISS")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(movie_ids))
    
    print(f"Индекс успешно создан и сохранен. Содержит {index.ntotal} фильмов")
    return index

def print_movie_info(movie, rank, score):
    print(f"{rank}. {movie.movie_name} ({movie.year})")
    print(f"Сходство: {score:.4f}")
    
    genres = list(movie.genre.all().values_list('name', flat=True))
    if genres:
        print(f"Жанры: {', '.join(genres)}")
    
    directors = list(movie.director.all().values_list('name', flat=True))
    if directors:
        print(f"Режиссер: {directors[0]}")
    
    if movie.overview:
        print(movie.overview[:400])

def calculate_metrics(retrieved_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    retrieved_k = retrieved_ids[:k]
    metrics = {}
    
    # Precision@k
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    metrics[f'P@{k}'] = relevant_retrieved / k if k > 0 else 0
    
    # Recall@k
    metrics[f'R@{k}'] = relevant_retrieved / len(relevant_set) if relevant_set else 0
    
    # F1@k
    p = metrics[f'P@{k}']
    r = metrics[f'R@{k}']
    metrics[f'F1@{k}'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    return metrics

def evaluate_model(model, movies_data, test_queries):
        
    # Конвертируем названия в ID
    for q in test_queries:
        q['relevant_ids'] = []
        for title in q['relevant_titles']:
            # Ищем похожие названия
            for movie in movies_data:
                if title.lower() in movie.movie_name.lower():
                    q['relevant_ids'].append(movie.id)
    
    # Собираем метрики
    all_metrics = defaultdict(list)
    detailed_results = []
    
    for q in test_queries:
        print(f"Запрос: '{q['query']}'")
        print(f"Релевантные фильмы: {q['relevant_titles']}")
        
        # Получаем результаты поиска
        retrieved_movies, scores = model(q['query'], top_k=20)
        retrieved_ids = [m.id for m in retrieved_movies]
        
        # Считаем метрики для разных k
        for k in [1, 3, 5, 10]:
            metrics = calculate_metrics(retrieved_ids, q['relevant_ids'], k)
            for name, value in metrics.items():
                all_metrics[f"{name}_{q['type']}"].append(value)
                all_metrics[name].append(value)
        
        # Сохраняем детальные результаты
        detailed_results.append({
            'query': q['query'],
            'type': q['type'],
            'relevant': q['relevant_titles'],
            'retrieved': [m.movie_name for m in retrieved_movies[:5]],
            'scores': list(scores[:5]),
        })
        
        # Показываем первые результаты
        print("Найденные фильмы:")
        for i, (movie, score) in enumerate(zip(retrieved_movies[:5], scores[:5]), 1):
            relevant = "Relevant" if movie.id in q['relevant_ids'] else "Not_Relevant"
            print(f"  {i}. [{relevant}] {movie.movie_name} ({movie.year}) - {score:.4f}")
    
    # Усредняем метрики
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        avg_metrics[metric_name] = np.mean(values)
    
    # Основные метрики
    print("Основные метрики:")
    for k in [1, 3, 5, 10]:
        print(f"  Precision@{k}: {avg_metrics.get(f'P@{k}', 0):.4f}")
        print(f"  Recall@{k}:    {avg_metrics.get(f'R@{k}', 0):.4f}")
        print(f"  F1@{k}:        {avg_metrics.get(f'F1@{k}', 0):.4f}")
    
    return avg_metrics, detailed_results

def test_single_query():
    query = input("Введите поисковый запрос: ").strip()
    
    if not query:
        print("Запрос не может быть пустым")
        return
    
    print(f"Поиск: '{query}'")
    
    movies, scores = search_movies(query, top_k=10)
    
    if movies:
        print(f"Найдено {len(movies)} фильмов:")
        for i, (movie, score) in enumerate(zip(movies, scores), 1):
            print_movie_info(movie, i, score)
    else:
        print("Ничего не найдено")

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myapp.settings")
    django.setup()

    from app.models import Movie
    from test_queries import TEST_QUERIES

    test_single_query()
    movies_data = list(Movie.objects.prefetch_related('genre', 'director', 'country').all())
    test_queries = TEST_QUERIES
    metrics, details = evaluate_model(search_movies, movies_data, test_queries)