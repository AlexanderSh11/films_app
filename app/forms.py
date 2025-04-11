from django import forms
from .models import MovieRating

class RatingForm(forms.ModelForm):
    class Meta:
        model = MovieRating
        fields = ['user_rating']
        widgets = {
            'user_rating': forms.Select(choices=[
                (None, '-- Оцените --'),
                (1, '★ (1)'), 
                (2, '★★ (2)'),
                (3, '★★★ (3)'),
                (4, '★★★★ (4)'),
                (5, '★★★★★ (5)')
            ], attrs={'class': 'rating-select'})
        }