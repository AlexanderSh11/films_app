from django.db import models
from django.contrib.auth.models import User

from myapp.constants import AGE_RATING_CHOICES, MOVIE_RATING_CHOICES, NAME_MAX_LENGTH, POSTER_UPLOAD_PATH


class Genre(models.Model):
    name = models.CharField(max_length=NAME_MAX_LENGTH, unique=True, verbose_name='Жанр')

    class Meta:
        db_table = 'genres'
        verbose_name = 'жанр'
        verbose_name_plural = 'Жанры'

    def __str__(self):
        return self.name


class Director(models.Model):
    name = models.CharField(max_length=NAME_MAX_LENGTH, unique=True, verbose_name='Режиссёр')

    class Meta:
        db_table = 'directors'
        verbose_name = 'режиссёр'
        verbose_name_plural = 'Режиссёры'

    def __str__(self):
        return self.name
    
class Country(models.Model):
    name = models.CharField(max_length=NAME_MAX_LENGTH, unique=True, verbose_name='Страна')

    class Meta:
        db_table = 'countries'
        verbose_name = 'страна'
        verbose_name_plural = 'Страны'

    def __str__(self):
        return self.name

class Movie(models.Model):
    id = models.AutoField(primary_key=True)
    poster_link = models.TextField(blank=True, null=True, verbose_name = 'Ссылка на постер')
    local_poster = models.ImageField(upload_to=POSTER_UPLOAD_PATH, blank=True, null=True, verbose_name = 'Локальная ссылка на постер')
    movie_name = models.TextField(blank=False, null=False, verbose_name = 'Название')
    year = models.IntegerField(blank=False, null=False, verbose_name = 'Год')
    runtime = models.TextField(blank=True, null=True, verbose_name = 'Продолжительность', help_text = 'Формат: "<число> минут(ы)"')
    rating = models.FloatField(blank=True, null=True, verbose_name = 'Оценка')
    overview = models.TextField(blank=True, null=True, verbose_name = 'Описание')
    meta_score = models.IntegerField(blank=True, null=True, verbose_name = 'Meta-оценка')
    age_rating = models.IntegerField(blank=True, null=True, choices=AGE_RATING_CHOICES, verbose_name = 'Возрастной рейтинг')
    
    genre = models.ManyToManyField('Genre', blank=True, verbose_name='Жанры')
    director = models.ManyToManyField('Director', blank=True, verbose_name='Режиссёры')
    country = models.ManyToManyField('Country', blank=True, verbose_name='Страны')
    
    @property
    def poster_display_url(self):
        """Возвращает правильный URL для отображения"""
        return f"/posters/posters/{self.id}.jpg"
    class Meta:
        managed = True
        db_table = 'movie'
        unique_together = (('movie_name', 'year'),)
        verbose_name = 'фильм'
        verbose_name_plural = 'Фильмы'
    def __str__(self):
        return f"{self.movie_name} {self.year}"

class Favorite(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'favorite'
        unique_together = (('user', 'movie'),)
        verbose_name = 'избранное'
        verbose_name_plural = 'Избранные'

    def __str__(self):
        return f"{self.user.username} - {self.movie.movie_name} ({self.movie.year})"
    
class MovieRating(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user_rating = models.PositiveIntegerField(
        blank=True, null=True, 
        choices=MOVIE_RATING_CHOICES
    )
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'rating'
        unique_together = (('user', 'movie'),)
        verbose_name = 'оценка'
        verbose_name_plural = 'Оценки'

    def __str__(self):
        return f"{self.user.username} - {self.movie.movie_name} ({self.movie.year}): {self.user_rating}"
