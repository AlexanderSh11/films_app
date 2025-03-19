from django.shortcuts import render
from django.views.generic import TemplateView
from .models import Movie
# Create your views here.
class HomePageView(TemplateView):
    template_name = 'home.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['movies'] = Movie.objects.all()[:100]
        return context