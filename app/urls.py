from django.urls import path
from .views import SearchPageView, HomePageView, add_to_favorites, favorites, movie_detail, remove_from_favorites

urlpatterns = [
    path('search/', SearchPageView.as_view(), name='search'),
    path('movie/<int:movie_id>/favorite/add/', add_to_favorites, name='add_to_favorites'),
    path('movie/<int:movie_id>/favorite/remove/', remove_from_favorites, name='remove_from_favorites'),
    path('movie/<int:movie_id>/', movie_detail, name='movie_detail'),
    path('favorites/<int:user_id>/', favorites, name='favorites'),
    path('', HomePageView.as_view(), name='home'),
]