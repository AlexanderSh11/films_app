from django.contrib import admin
from app.models import Movie, Favorite, MovieRating

# admin (postgres)
# kinoman (postgres)
admin.site.register(Movie)
admin.site.register(Favorite)
admin.site.register(MovieRating)