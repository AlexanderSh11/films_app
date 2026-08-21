from django.urls import path

from accounts import views


urlpatterns = [
    path("signup/", views.register, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    path("profile/<int:user_id>/", views.profile, name="profile"),
]
