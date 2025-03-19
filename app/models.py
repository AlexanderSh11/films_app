from django.db import models

# Create your models here.
class Movie(models.Model):
    name = models.TextField(primary_key=True)  # The composite primary key (name, year) found, that is not supported. The first column is selected.
    rating = models.TextField(blank=True, null=True)
    genre = models.TextField(blank=True, null=True)
    year = models.IntegerField()
    released = models.TextField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    votes = models.FloatField(blank=True, null=True)
    director = models.TextField(blank=True, null=True)
    writer = models.TextField(blank=True, null=True)
    star = models.TextField(blank=True, null=True)
    country = models.TextField(blank=True, null=True)
    budget = models.FloatField(blank=True, null=True)
    gross = models.FloatField(blank=True, null=True)
    company = models.TextField(blank=True, null=True)
    runtime = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'movies'
        unique_together = (('name', 'year'),)
    def __str__(self):
        return f"{self.name} {self.year}"

class Favorite(models.Model):
    user_id = models.IntegerField(primary_key=True)
    movie_name = models.TextField()
    movie_year = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'favorites'
        unique_together = (('user_id', 'movie_name', 'movie_year'),)