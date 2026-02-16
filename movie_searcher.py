import os
import django
import sys
from pathlib import Path

# Настройка Django окружения
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myapp.settings")
django.setup()

from app.models import Movie, AGE_RATINGS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
        age_text = dict(AGE_RATINGS).get(movie.age_rating, f"{movie.age_rating}+")
        parts.append(f"возрастное ограничение: {age_text}")
    
    return ". ".join(parts)

def build_faiss_index(force_rebuild=False):
    """Создает индекс FAISS для всех фильмов"""
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

def search_movies(query, top_k=5, model=None, index=None, movie_ids=None):
    """Ищет фильмы по запросу"""
    
    # Загружаем модель если не передана
    if model is None:
        model = SentenceTransformer('cointegrated/rubert-tiny2')
    
    # Загружаем индекс если не передан
    if index is None:
        index = faiss.read_index('movie_index.faiss')
    if movie_ids is None:
        movie_ids = np.load('movie_ids.npy')
    
    # Создаем эмбеддинг из запроса и ищем похожие
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Получаем ID фильмов
    result_ids = [movie_ids[i] for i in indices[0]]
    movies = Movie.objects.filter(id__in=result_ids).prefetch_related('genre', 'director', 'country')
    
    # Сортируем по порядку из индекса
    movie_dict = {movie.id: movie for movie in movies}
    ordered_movies = []
    for mid in result_ids:
        if mid in movie_dict:
            ordered_movies.append(movie_dict[mid])
    
    return ordered_movies, distances[0]

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
    
    # Загружаем модель
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    
    index_path = 'movie_index.faiss'
    ids_path = 'movie_ids.npy'
    
    if not os.path.exists(index_path) or not os.path.exists(ids_path):
        print("Индекс не найден. Создаём новый")
        build_faiss_index(force_rebuild=True)
    
    # Загружаем 
    index = faiss.read_index(index_path)
    movie_ids = np.load(ids_path)
    
    movies, scores = search_movies(query, top_k=5, model=model, index=index, movie_ids=movie_ids)
    
    if movies:
        print(f"Найдено {len(movies)} фильмов:")
        for i, (movie, score) in enumerate(zip(movies, scores), 1):
            print_movie_info(movie, i, score)
    else:
        print("Ничего не найдено")

if __name__ == "__main__":
    test_single_query()