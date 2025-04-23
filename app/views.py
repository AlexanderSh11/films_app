from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.urls import reverse
import requests
from django.views.generic import TemplateView

from app.forms import RatingForm
from content_based import MovieRecommender
from .models import Movie, Favorite, MovieRating
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
        if sort == 'названию':
            movies = movies.order_by('movie_name', 'year')
        elif sort == 'рейтингу (сначала лучшие)':
            movies = movies.filter(rating__isnull=False).order_by('-rating')
        elif sort == 'рейтингу (сначала худшие)':
            movies = movies.filter(rating__isnull=False).order_by('rating')
        elif sort == 'году выхода (сначала новые)':
            movies = movies.order_by('-year', 'movie_name')
        elif sort == 'году выхода (сначала старые)':
            movies = movies.order_by('year', 'movie_name')

        user_favorites_ids = []
        if request.user.is_authenticated:
            user_favorites_ids = Favorite.objects.select_related('movie').filter(user_id=request.user.id).values_list('movie_id', flat=True)

        context['selected_genre'] = genre
        context['selected_sort'] = sort
        context['movies'] = movies[:100]
        context['user_favorites'] = user_favorites_ids
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

        user_favorites_ids = []
        if request.user.is_authenticated:
            user_favorites_ids = Favorite.objects.select_related('movie').filter(user_id=request.user.id).values_list('movie_id', flat=True)
               
            recommender = MovieRecommender(request.user)
            recommendations = recommender.recommend_movies(10)
        else:
            recommendations = []
        context['movies_by_genre'] = movies_by_genre
        context['user_favorites'] = user_favorites_ids
        context['recommendations'] = recommendations
        context['show_recommendations'] = recommendations
        return context
    

def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    is_favorite = False
    user_rating = 'Нет оценки'
    if request.user.is_authenticated:
        fav = Favorite.objects.filter(user=request.user, movie=movie)
        is_favorite = fav.exists()
        rate = MovieRating.objects.filter(user=request.user, movie=movie)
        if rate.exists():
            user_rating = rate[0].user_rating
    return render(request, 'movie_detail.html', {'movie': movie, 'is_favorite': is_favorite, 'user_rating': user_rating})

@login_required
def favorites(request, user_id):
    user = get_object_or_404(User, id=user_id)
    favorites = Favorite.objects.select_related('movie').filter(user_id=user_id)
    return render(request, 'favorites.html', {'user': user, 'favorites': favorites})

@login_required
def add_to_favorites(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    Favorite.objects.get_or_create(user=request.user, movie=movie)
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('home')))

@login_required
def remove_from_favorites(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    Favorite.objects.filter(user=request.user, movie=movie).delete()
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', reverse('home')))

@login_required
def rate_movie(request, movie_id):
    movie = get_object_or_404(Movie, pk=movie_id)
    
    # Получаем или создаем запись в избранном
    rating, created = MovieRating.objects.get_or_create(
        user=request.user,
        movie=movie
    )
    
    if request.method == 'POST':
        form = RatingForm(request.POST, instance=rating)
        if form.is_valid():
            form.save()
            return redirect('movie_detail', movie_id=movie.id)
    else:
        form = RatingForm(instance=rating)
    
    return redirect('movie_detail', movie_id=movie.id)