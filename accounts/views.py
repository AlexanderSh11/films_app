from django.shortcuts import redirect, render
from app.models import Favorite, Movie
from django.contrib.auth.models import User
from .forms import CustomUserCreationForm
from django.contrib.auth.views import LogoutView as AuthLogoutView
from django.urls import reverse_lazy
from django.views import generic

class Logout(AuthLogoutView):
    next_page = reverse_lazy('home')

def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})