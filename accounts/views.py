from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import redirect, render, get_object_or_404

from app.models import Favorite, MovieRating
from accounts.forms import CustomUserCreationForm
from friends.models import Friendship


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
    profile_user = get_object_or_404(User, id=user_id)
    favorites = Favorite.objects.filter(user=profile_user).select_related('movie')
    favorites_count = favorites.count()
    ratings = MovieRating.objects.filter(user=profile_user).select_related('movie')
    ratings_count = ratings.count()
    user_favorites_ids = Favorite.objects.filter(user=profile_user).values_list('movie_id', flat=True)
    friendships = Friendship.objects.filter(Q(from_user=profile_user) | Q(to_user=profile_user)).select_related('from_user', 'to_user')
    friends = []
    for friendship in friendships:
        if friendship.from_user == profile_user:
            friends.append(friendship.to_user)
        else:
            friends.append(friendship.from_user)
    is_friend = False
    if request.user.is_authenticated and request.user != profile_user:
        is_friend = Friendship.objects.filter((Q(from_user=request.user, to_user=profile_user) | Q(from_user=profile_user, to_user=request.user))).exists()

    context = {
        'user': profile_user,
        'current_user': request.user,
        'favorites': favorites,
        'ratings': ratings,
        'favorites_count': favorites_count,
        'ratings_count': ratings_count,
        'user_favorites_ids': user_favorites_ids,
        'friends': friends,
        'is_friend': is_friend,
        'is_own_profile': request.user == profile_user,
    }
    
    return render(request, 'registration/profile.html', context)
