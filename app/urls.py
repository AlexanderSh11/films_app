from django.urls import path
from .views import SearchPageView, HomePageView, add_to_favorites, favorites, ratings, movie_detail, rate_movie, remove_from_favorites

urlpatterns = [
    path('search/', SearchPageView.as_view(), name='search'),
    path('movie/<int:movie_id>/favorite/add/', add_to_favorites, name='add_to_favorites'),
    path('movie/<int:movie_id>/favorite/remove/', remove_from_favorites, name='remove_from_favorites'),
    path('movie/<int:movie_id>/', movie_detail, name='movie_detail'),
    path('favorites/<int:user_id>/', favorites, name='favorites'),
    path('ratings/<int:user_id>/', ratings, name='ratings'),
    path('movie/<int:movie_id>/rate/', rate_movie, name='rate_movie'),
    path('', HomePageView.as_view(), name='home'),
]