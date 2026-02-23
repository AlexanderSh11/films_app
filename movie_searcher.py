import os
import django
from django.core.cache import cache
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Глобальные переменные для кэширования
_MODEL = None
_INDEX = None
_ALL_MOVIE_IDS = None

def load_model_and_index():
    """Загружает модель и индекс при первом вызове"""
    global _MODEL, _INDEX, _ALL_MOVIE_IDS, _ALL_EMBEDDINGS, _ALL_EMBEDDINGS_DICT

    _MODEL = cache.get('semantic_model')
    _INDEX = cache.get('semantic_index')
    _ALL_MOVIE_IDS = cache.get('semantic_ids')

    if _MODEL is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_model\content\drive\MyDrive\Films app\\fine_tuned_rubert_tiny2')
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

    embeddings_path = 'movie_embeddings.npy'
    if os.path.exists(embeddings_path):
        _ALL_EMBEDDINGS = np.load(embeddings_path)
        # Создаем словарь {movie_id: embedding}
        _ALL_EMBEDDINGS_DICT = {
            movie_id: _ALL_EMBEDDINGS[i] 
            for i, movie_id in enumerate(_ALL_MOVIE_IDS)
        }
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
    parts.append(f"год: {movie.year}")
    
    genres = list(movie.genre.all().values_list('name', flat=True))
    if genres:
        parts.append(f"жанры: {', '.join(genres)}")
    
    directors = list(movie.director.all().values_list('name', flat=True))
    if directors:
        parts.append(f"режиссёр: {', '.join(directors)}")
    
    countries = list(movie.country.all().values_list('name', flat=True))
    if countries:
        parts.append(f"страна: {', '.join(countries)}")
    
    if movie.overview:
        parts.append(f"описание: {movie.overview}")
    
    if movie.runtime:
        parts.append(f"длительность: {movie.runtime}")
    
    if movie.rating:
        parts.append(f"рейтинг: {movie.rating} из 10")
    
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
    
    # Если передан список конкретных ID для поиска
    if movie_ids is not None:
        # Получаем фильмы из БД для указанных ID
        movies_subset = Movie.objects.filter(id__in=movie_ids).prefetch_related('genre', 'director', 'country')
        
        if not movies_subset:
            return [], []
        
        # Создаем тексты и эмбеддинги для подмножества
        embeddings = []
        valid_movie_ids = []
        
        for mid in movie_ids:
            if mid in _ALL_EMBEDDINGS_DICT:
                embeddings.append(_ALL_EMBEDDINGS_DICT[mid])
                valid_movie_ids.append(mid)
        
        if not embeddings:
            return [], []
        
        # Преобразуем в numpy array
        embeddings = np.array(embeddings)
        
        # Создаем временный индекс
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(embeddings)
        
        # Ищем
        distances, indices = temp_index.search(query_embedding, min(top_k, len(movies_subset)))
        
        # Получаем результаты
        movies_list = list(movies_subset)
        result_ids = [movies_list[i].id for i in indices[0]]
        
        # Возвращаем объекты фильмов в правильном порядке
        result_movies = []
        for movie_id in result_ids:
            for movie in movies_list:
                if movie.id == movie_id:
                    result_movies.append(movie)
                    break
        
        return result_movies, distances[0]
    
    # Поиск по всему индексу
    distances, indices = _INDEX.search(query_embedding, top_k)
    
    # Получаем ID фильмов
    result_ids = [_ALL_MOVIE_IDS[i] for i in indices[0]]
    
    # Получаем фильмы из БД
    movies = Movie.objects.filter(id__in=result_ids).prefetch_related('genre', 'director', 'country')
    
    # Сортируем по порядку из индекса
    movie_dict = {movie.id: movie for movie in movies}
    ordered_movies = [movie_dict[mid] for mid in result_ids if mid in movie_dict]
    
    # увеличиваем score для фильмов с совпадением по названию
    query_lower = query.lower()
    boosted_distances = list(distances[0].copy())
    
    for i, movie in enumerate(ordered_movies):
        title_lower = movie.movie_name.lower()
        
        # Если название полностью совпадает с запросом
        if query_lower == title_lower:
            boosted_distances[i] *= 0.6
        # Если запрос является частью названия или название часть запроса
        elif query_lower in title_lower or title_lower in query_lower:
            boosted_distances[i] *= 0.7

    # Сортируем заново с учетом score
    sorted_indices = np.argsort(boosted_distances)
    ordered_movies = [ordered_movies[i] for i in sorted_indices]
    boosted_distances = [boosted_distances[i] for i in sorted_indices]

    return ordered_movies, boosted_distances

def build_faiss_index(force_rebuild=False):
    """Создает индекс FAISS для всех фильмов"""
    from app.models import Movie
    index_path = 'movie_index.faiss'
    ids_path = 'movie_ids.npy'
    
    # Проверяем существует ли уже индекс
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(ids_path):
        return
    
    # Загружаем модель cointegrated/rubert-tiny2
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    
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

    test_single_query()