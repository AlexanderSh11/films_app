from django.contrib import admin
from friends.models import Friendship


@admin.register(Friendship)
class FriendshipAdmin(admin.ModelAdmin):
    list_display = ("id", "from_user", "to_user", "created_at")
    list_filter = ("created_at",)
    search_fields = ("from_user__username", "to_user__username")
    readonly_fields = ("created_at",)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("from_user", "to_user")
