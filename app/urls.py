from django.urls import path
from .views import SearchPageView, HomePageView

urlpatterns = [
    path('search/', SearchPageView.as_view(), name='search'),
    path('', HomePageView.as_view(), name='home'),
]