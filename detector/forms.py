# detector/forms.py
from django import forms
from .models import Violation, Camera

class ViolationReviewForm(forms.ModelForm):
    class Meta:
        model = Violation
        fields = ['status', 'notes']
        widgets = {
            'status': forms.Select(attrs={'class': 'form-select'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
        }

class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = ['name', 'location', 'rtsp_url', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'rtsp_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'rtsp://...'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }