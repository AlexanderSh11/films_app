from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.db.models import Q
from friends.models import Friendship

User = get_user_model()

@login_required
def add_friend(request, user_id):
    friend = get_object_or_404(User, id=user_id)
    
    # Нельзя добавить самого себя
    if request.user == friend:
        messages.error(request, 'Нельзя добавить самого себя в друзья')
        return redirect('profile', user_id=user_id)
    
    # Проверяем, не друзья ли уже
    existing = Friendship.objects.filter(
        Q(from_user=request.user, to_user=friend) |
        Q(from_user=friend, to_user=request.user)
    ).first()
    
    if existing:
        messages.info(request, f'Вы уже друзья с {friend.username}')
    else:
        # Создаем запись о дружбе (всегда from_user=текущий пользователь)
        Friendship.objects.create(from_user=request.user, to_user=friend)
        messages.success(request, f'{friend.username} добавлен в друзья')
    
    return redirect('profile', user_id=user_id)

@login_required
def remove_friend(request, user_id):
    friend = get_object_or_404(User, id=user_id)
    
    # Находим и удаляем запись о дружбе в любом направлении
    friendship = Friendship.objects.filter(
        Q(from_user=request.user, to_user=friend) |
        Q(from_user=friend, to_user=request.user)
    ).first()
    
    if friendship:
        friendship.delete()
        messages.success(request, f'{friend.username} удален из друзей')
    else:
        messages.error(request, 'Ошибка при удалении из друзей')
    
    return redirect('profile', user_id=user_id)
