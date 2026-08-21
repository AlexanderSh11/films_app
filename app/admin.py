from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html

from app.models import Genre, Director, Country, Movie, Favorite, MovieRating


class CustomUserAdmin(UserAdmin):
    list_display = UserAdmin.list_display + (
        "last_login",
        "date_joined",
        "favorites_count",
        "ratings_count",
    )

    list_filter = UserAdmin.list_filter + ("is_active", "is_staff", "date_joined")

    search_fields = UserAdmin.search_fields + ("email", "first_name", "last_name")

    fieldsets = UserAdmin.fieldsets + (
        (
            "Статистика",
            {
                "fields": ("favorites_count", "ratings_count"),
            },
        ),
    )

    readonly_fields = UserAdmin.readonly_fields + ("favorites_count", "ratings_count")

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .prefetch_related("favorite_set", "movierating_set")
        )

    @admin.display(description="Избранное")
    def favorites_count(self, obj):
        count = obj.favorite_set.count()
        return format_html(
            '<span style="color: {};">{}</span>',
            "green" if count > 0 else "gray",
            count,
        )

    @admin.display(description="Оценки")
    def ratings_count(self, obj):
        count = obj.movierating_set.count()
        return format_html(
            '<span style="color: {};">{}</span>', "blue" if count > 0 else "gray", count
        )


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "movie_count"]
    list_display_links = ["id", "name"]
    search_fields = ["name"]
    list_filter = ["name"]
    ordering = ["name"]

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("movie_set")

    @admin.display(description="Фильмов")
    def movie_count(self, obj):
        return obj.movie_set.count()


@admin.register(Director)
class DirectorAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "movie_count"]
    list_display_links = ["id", "name"]
    search_fields = ["name"]
    list_filter = ["name"]
    ordering = ["name"]

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("movie_set")

    @admin.display(description="Фильмов")
    def movie_count(self, obj):
        return obj.movie_set.count()


@admin.register(Country)
class CountryAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "movie_count"]
    list_display_links = ["id", "name"]
    search_fields = ["name"]
    list_filter = ["name"]
    ordering = ["name"]

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("movie_set")

    @admin.display(description="Фильмов")
    def movie_count(self, obj):
        return obj.movie_set.count()


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "movie_name",
        "year",
        "rating",
        "age_rating",
        "favorites_count",
        "ratings_count",
    ]
    list_display_links = ["id", "movie_name"]

    search_fields = ["movie_name", "overview", "genre__name", "director__name"]
    list_filter = ["year", "rating", "age_rating", "genre", "director"]
    ordering = ["-year", "movie_name"]
    sortable_by = ["id", "movie_name", "year", "rating"]

    filter_horizontal = ["genre", "director", "country"]
    readonly_fields = ["favorites_count", "ratings_count"]
    list_per_page = 25

    fieldsets = (
        (
            "Основная информация",
            {
                "fields": (
                    ("movie_name", "year"),
                    ("rating", "meta_score"),
                    ("runtime", "age_rating"),
                    "overview",
                ),
            },
        ),
        (
            "Медиа",
            {
                "fields": (("poster_link", "local_poster"),),
            },
        ),
        (
            "Связанные записи",
            {
                "fields": ("genre", "director", "country"),
            },
        ),
        (
            "Служебная информация",
            {
                "fields": ("favorites_count", "ratings_count"),
                "classes": ("collapse",),
            },
        ),
    )

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .prefetch_related("favorite_set", "movierating_set")
        )

    @admin.display(description="В избранном", ordering="favorites_count")
    def favorites_count(self, obj):
        return obj.favorite_set.count()

    @admin.display(description="Оценок", ordering="ratings_count")
    def ratings_count(self, obj):
        return obj.movierating_set.count()


@admin.register(Favorite)
class FavoriteAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "movie", "added_at"]
    list_display_links = ["id", "user"]
    search_fields = ["user__username", "movie__movie_name"]
    list_filter = ["added_at"]
    ordering = ["-added_at"]
    raw_id_fields = ["user", "movie"]
    list_per_page = 50


@admin.register(MovieRating)
class MovieRatingAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "movie", "user_rating", "added_at"]
    list_display_links = ["id", "user"]
    search_fields = ["user__username", "movie__movie_name", "user_rating"]
    list_filter = ["user_rating", "added_at"]
    ordering = ["-added_at"]
    raw_id_fields = ["user", "movie"]
    list_per_page = 50
