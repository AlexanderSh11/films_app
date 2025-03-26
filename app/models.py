from django.db import models

# Create your models here.
class Movie(models.Model):
    poster_link = models.TextField(blank=True, null=True)
    movie_name = models.TextField(primary_key=True)  # The composite primary key (name, year) found, that is not supported. The first column is selected.
    year = models.TextField()
    certificate = models.TextField(blank=True, null=True)
    runtime = models.TextField(blank=True, null=True)
    genre = models.TextField(blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    overview = models.TextField(blank=True, null=True)
    meta_score = models.IntegerField(blank=True, null=True)
    director = models.TextField(blank=True, null=True)
    star1 = models.TextField(blank=True, null=True)
    star2 = models.TextField(blank=True, null=True)
    star3 = models.TextField(blank=True, null=True)
    star4 = models.TextField(blank=True, null=True)
    votes = models.BigIntegerField(blank=True, null=True)
    gross = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'movie'
        unique_together = (('movie_name', 'year'),)
    def __str__(self):
        return f"{self.movie_name} {self.year}"

class Favorite(models.Model):
    user_id = models.IntegerField(primary_key=True)
    movie_name = models.TextField()
    movie_year = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'favorites'
        unique_together = (('user_id', 'movie_name', 'movie_year'),)