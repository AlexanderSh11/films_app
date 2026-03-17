from django.urls import path
from friends.views import add_friend, remove_friend

app_name = 'friends'

urlpatterns = [
    path('add/<int:user_id>/', add_friend, name='add_friend'),
    path('remove/<int:user_id>/', remove_friend, name='remove_friend'),
]
