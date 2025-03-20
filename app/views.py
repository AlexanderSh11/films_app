from django.shortcuts import render
from django.views.generic import TemplateView
from .models import Movie
# Create your views here.
class HomePageView(TemplateView):
    template_name = 'home.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        genre = request.GET.get('genre', 'All')
        sort = request.GET.get('sort', 'year')
        if genre != 'All':
            movies = Movie.objects.filter(genre=genre)
        else:
            movies = Movie.objects.all()
        if sort == 'title':
            movies = movies.order_by('name', 'year')
        elif sort == 'highest score':
            movies = movies.filter(score__isnull=False).order_by('-score')
        elif sort == 'lowest score':
            movies = movies.filter(score__isnull=False).order_by('score')
        elif sort == 'year':
            movies = movies.order_by('year', 'name')
        context['selected_genre'] = genre
        context['selected_sort'] = sort
        context['movies'] = movies[:100]
        return context