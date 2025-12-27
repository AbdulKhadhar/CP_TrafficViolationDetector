# detector/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('cameras/', views.cameras_list, name='cameras_list'),
    path('violations/', views.violations_list, name='violations_list'),
    path('violations/<int:pk>/', views.violation_detail, name='violation_detail'),
    path('upload/', views.upload_video, name='upload_video'),
    path('camera/<int:camera_id>/start/', views.start_live_detection, name='start_detection'),
    path('camera/<int:camera_id>/stop/', views.stop_live_detection, name='stop_detection'),
    path('camera/<int:camera_id>/feed/', views.live_feed, name='live_feed'),
    path('analytics/', views.analytics, name='analytics'),
    path('api/violations/', views.api_violations, name='api_violations'),
]