import requests
from django.shortcuts import render
from django.views.generic import TemplateView
from .models import Movie
# Create your views here.
class SearchPageView(TemplateView):
    template_name = 'search.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        genre = request.GET.get('genre', 'All')
        sort = request.GET.get('sort', 'year')
        movies = Movie.objects.all()
        for movie in movies:
            movie.genre_list = movie.genre.split(", ")
        if genre != 'All':
            movies = Movie.objects.filter(genre__icontains=genre)
        if sort == 'title':
            movies = movies.order_by('movie_name', 'year')
        elif sort == 'highest rating':
            movies = movies.filter(rating__isnull=False).order_by('-rating')
        elif sort == 'lowest rating':
            movies = movies.filter(rating__isnull=False).order_by('rating')
        elif sort == 'newest':
            movies = movies.order_by('-year', 'movie_name')
        elif sort == 'oldest':
            movies = movies.order_by('year', 'movie_name')
        context['selected_genre'] = genre
        context['selected_sort'] = sort
        context['movies'] = movies[:100]
        return context

class HomePageView(TemplateView):
    template_name = 'home.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        movies_by_genre = {}

        movies = Movie.objects.all()
        for movie in movies:
            movie_genres = movie.genre.split(", ")
            for movie_genre in movie_genres:
                if movie_genre not in movies_by_genre:
                    movies_by_genre[movie_genre] = []
                movies_by_genre[movie_genre].append(movie)

        context['movies_by_genre'] = movies_by_genre
        return context