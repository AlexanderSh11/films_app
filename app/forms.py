from django import forms

from app.models import MovieRating
from myapp.constants import RATING_CHOICES


class RatingForm(forms.ModelForm):
    class Meta:
        model = MovieRating
        fields = ['user_rating']
        widgets = {
            'user_rating': forms.RadioSelect(choices=RATING_CHOICES)
        }
