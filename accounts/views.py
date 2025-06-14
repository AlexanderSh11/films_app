from django.shortcuts import redirect, render, get_object_or_404
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
    return render(request, 'registration/profile.html', {'user': user})