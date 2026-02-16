from django.apps import AppConfig
from movie_searcher import load_model_and_index

load_model_and_index()

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    verbose_name = 'Основное приложение'
