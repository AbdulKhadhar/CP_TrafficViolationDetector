from django.contrib import admin
from .models import Camera, Violation, DetectionLog, SystemSettings

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'is_active', 'created_at']
    list_filter = ['is_active']
    search_fields = ['name', 'location']

@admin.register(Violation)
class ViolationAdmin(admin.ModelAdmin):
    list_display = ['id', 'violation_type', 'camera', 'detected_at', 'confidence_score', 'status']
    list_filter = ['violation_type', 'status', 'vehicle_type', 'detected_at']
    search_fields = ['license_plate']
    date_hierarchy = 'detected_at'

@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ['camera', 'started_at', 'ended_at', 'frames_processed', 'violations_detected', 'is_running']
    list_filter = ['is_running', 'camera']

@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ['detection_confidence_threshold', 'helmet_detection_enabled', 'plate_detection_enabled']