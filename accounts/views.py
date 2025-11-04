from django.shortcuts import redirect, render, get_object_or_404

from app.models import Favorite, MovieRating
from .forms import CustomUserCreationForm
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

def logout_view(request):
    if request.user.is_authenticated:
        logout(request)
    return redirect('home')

def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


@login_required
def profile(request, user_id):
    user = get_object_or_404(User, id=user_id)
    favorites = Favorite.objects.filter(user=user).select_related('movie')
    favorites_count = favorites.count()
    ratings = MovieRating.objects.filter(user=user).select_related('movie')
    ratings_count = ratings.count()
    user_favorites_ids = Favorite.objects.filter(user=user).values_list('movie_id', flat=True)
    
    context = {
        'user': user,
        'favorites': favorites,
        'ratings': ratings,
        'favorites_count': favorites_count,
        'ratings_count': ratings_count,
        'user_favorites_ids': user_favorites_ids,
    }
    
    return render(request, 'registration/profile.html', context)