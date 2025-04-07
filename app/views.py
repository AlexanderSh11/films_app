from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
import requests
from django.views.generic import TemplateView
from .models import Movie, Favorite
from django.contrib.auth.models import User
# Create your views here.
class SearchPageView(TemplateView):
    template_name = 'search.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        genre = request.GET.get('genre', 'Все')
        sort = request.GET.get('sort', 'newest')
        movies = Movie.objects.all()
        for movie in movies:
            movie.genre_list = movie.genre.split(", ")
        if genre != 'Все':
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
    

def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    is_favorite = False
    if request.user.is_authenticated:
        is_favorite = Favorite.objects.filter(user=request.user, movie=movie).exists()
    return render(request, 'movie_detail.html', {'movie': movie, 'is_favorite': is_favorite})

@login_required
def add_to_favorites(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    Favorite.objects.get_or_create(user=request.user, movie=movie)
    return redirect('movie_detail', movie_id=movie_id)

@login_required
def remove_from_favorites(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    Favorite.objects.filter(user=request.user, movie=movie).delete()
    return redirect('movie_detail', movie_id=movie_id)