from collections import defaultdict
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.views.generic import TemplateView

from app.forms import RatingForm
from content_based import MovieRecommender
from .models import Movie, Favorite, MovieRating
from django.contrib.auth.models import User


class SearchPageView(TemplateView):
    template_name = 'search.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        genre = request.GET.get('genre', 'Все')
        title = request.GET.get('title', '')
        sort = request.GET.get('sort', 'newest')

        movies = Movie.objects.prefetch_related('genre').all()

        # Преобразуем ManyToManyField в список имен
        for movie in movies:
            movie.genre_list = [g.name for g in movie.genre.all()]

        if genre != 'Все':
            movies = movies.filter(genre__name__icontains=genre)

        if title:
            movies = movies.filter(movie_name__icontains=title)

        # Сортировка
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
            user_favorites_ids = Favorite.objects.filter(user=request.user).values_list('movie_id', flat=True)

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

        movies = Movie.objects.prefetch_related('genre').all()

        movies_by_genre = defaultdict(list)
        for movie in movies:
            for g in movie.genre.all():
                if len(movies_by_genre[g.name]) < 10:
                    movies_by_genre[g.name].append(movie)

        user_favorites_ids = []
        recommendations = []
        if request.user.is_authenticated:
            user_favorites_ids = Favorite.objects.filter(user=request.user).values_list('movie_id', flat=True)
            recommender = MovieRecommender(request.user)
            recommendations = recommender.recommend_movies(10)

        context['movies_by_genre'] = dict(movies_by_genre)
        context['user_favorites'] = user_favorites_ids
        context['recommendations'] = recommendations
        context['show_recommendations'] = recommendations
        return context


def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    is_favorite = False
    user_rating = 'Нет оценки'

    if request.user.is_authenticated:
        is_favorite = Favorite.objects.filter(user=request.user, movie=movie).exists()
        rate = MovieRating.objects.filter(user=request.user, movie=movie).first()
        if rate:
            user_rating = rate.user_rating

    # Преобразуем ManyToMany в списки для шаблона
    movie.genre_list = [g.name for g in movie.genre.all()]
    movie.director_list = [d.name for d in movie.director.all()]
    movie.country_list = [c.name for c in movie.country.all()]

    return render(request, 'movie_detail.html', {
        'movie': movie,
        'is_favorite': is_favorite,
        'user_rating': user_rating
    })


@login_required
def favorites(request, user_id):
    user = get_object_or_404(User, id=user_id)
    favorites = Favorite.objects.filter(user=user)
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
    rating, created = MovieRating.objects.get_or_create(user=request.user, movie=movie)

    if request.method == 'POST':
        form = RatingForm(request.POST, instance=rating)
        if form.is_valid():
            form.save()
            return redirect('movie_detail', movie_id=movie.id)
    return redirect('movie_detail', movie_id=movie.id)