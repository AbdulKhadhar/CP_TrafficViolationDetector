from django.db import models
from django.utils import timezone
import os

class Camera(models.Model):
    """Model for traffic cameras"""
    name = models.CharField(max_length=200)
    location = models.CharField(max_length=300)
    rtsp_url = models.URLField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.location}"

class Violation(models.Model):
    """Model for storing detected violations"""
    VIOLATION_TYPES = [
        ('NO_HELMET', 'No Helmet'),
        ('NO_PLATE', 'No Number Plate'),
        ('BOTH', 'No Helmet and No Plate'),
    ]
    
    VEHICLE_TYPES = [
        ('MOTORCYCLE', 'Motorcycle'),
        ('CAR', 'Car'),
        ('TRUCK', 'Truck'),
        ('BUS', 'Bus'),
        ('OTHER', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending Review'),
        ('CONFIRMED', 'Confirmed'),
        ('FALSE_POSITIVE', 'False Positive'),
        ('PROCESSED', 'Processed'),
    ]
    
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='violations')
    violation_type = models.CharField(max_length=20, choices=VIOLATION_TYPES)
    vehicle_type = models.CharField(max_length=20, choices=VEHICLE_TYPES)
    detected_at = models.DateTimeField(default=timezone.now)
    confidence_score = models.FloatField(help_text="AI confidence level (0-1)")
    
    # Image evidence
    image = models.ImageField(upload_to='violations/%Y/%m/%d/')
    annotated_image = models.ImageField(upload_to='violations/annotated/%Y/%m/%d/', blank=True)
    
    # Vehicle details (if detected)
    license_plate = models.CharField(max_length=50, blank=True, null=True)
    plate_confidence = models.FloatField(blank=True, null=True)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    reviewed_by = models.CharField(max_length=100, blank=True, null=True)
    reviewed_at = models.DateTimeField(blank=True, null=True)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-detected_at']
        indexes = [
            models.Index(fields=['-detected_at']),
            models.Index(fields=['status']),
            models.Index(fields=['violation_type']),
        ]
    
    def __str__(self):
        return f"{self.get_violation_type_display()} - {self.detected_at.strftime('%Y-%m-%d %H:%M')}"

class DetectionLog(models.Model):
    """Model for logging detection sessions"""
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(blank=True, null=True)
    frames_processed = models.IntegerField(default=0)
    violations_detected = models.IntegerField(default=0)
    is_running = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Detection Log - {self.camera.name} - {self.started_at.strftime('%Y-%m-%d %H:%M')}"

class SystemSettings(models.Model):
    """Model for system configuration"""
    detection_confidence_threshold = models.FloatField(default=0.5, help_text="Minimum confidence for violation detection")
    helmet_detection_enabled = models.BooleanField(default=True)
    plate_detection_enabled = models.BooleanField(default=True)
    frame_skip = models.IntegerField(default=5, help_text="Process every Nth frame")
    max_violations_per_minute = models.IntegerField(default=10, help_text="Rate limiting")
    notification_email = models.EmailField(blank=True)
    
    class Meta:
        verbose_name_plural = "System Settings"
    
    def __str__(self):
        return "System Configuration"