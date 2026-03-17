from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Friendship(models.Model):
    from_user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='friends_from',
        verbose_name='пользователь'
    )
    to_user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='friends_to',
        verbose_name='друг'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('from_user', 'to_user')
        verbose_name = 'дружественная связь'
        verbose_name_plural = 'Дружественные связи'

    def __str__(self):
        return f"{self.from_user} -> {self.to_user}"