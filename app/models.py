from datetime import timezone
from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Movie(models.Model):
    id = models.AutoField(primary_key=True)
    poster_link = models.TextField(blank=True, null=True)
    movie_name = models.TextField()
    year = models.IntegerField()
    runtime = models.TextField(blank=True, null=True)
    genre = models.TextField(blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    overview = models.TextField(blank=True, null=True)
    meta_score = models.IntegerField(blank=True, null=True)
    director = models.TextField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'movie'
        unique_together = (('movie_name', 'year'),)
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

    def __str__(self):
        return f"{self.user.username} - {self.movie.movie_name} ({self.movie.year})"