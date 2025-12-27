# detector/tests.py
from django.test import TestCase
from django.utils import timezone
from .models import Camera, Violation
import cv2
import numpy as np

class CameraModelTest(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(
            name="Test Camera",
            location="Test Location",
            is_active=True
        )
    
    def test_camera_creation(self):
        self.assertEqual(self.camera.name, "Test Camera")
        self.assertTrue(self.camera.is_active)
    
    def test_camera_string_representation(self):
        expected = "Test Camera - Test Location"
        self.assertEqual(str(self.camera), expected)

class ViolationModelTest(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(
            name="Test Camera",
            location="Test Location"
        )
        self.violation = Violation.objects.create(
            camera=self.camera,
            violation_type='NO_HELMET',
            vehicle_type='MOTORCYCLE',
            confidence_score=0.85,
            status='PENDING'
        )
    
    def test_violation_creation(self):
        self.assertEqual(self.violation.violation_type, 'NO_HELMET')
        self.assertEqual(self.violation.confidence_score, 0.85)
        self.assertEqual(self.violation.status, 'PENDING')
    
    def test_violation_status_choices(self):
        self.violation.status = 'CONFIRMED'
        self.violation.save()
        self.assertEqual(self.violation.status, 'CONFIRMED')

class DetectorTest(TestCase):
    def test_helmet_detector_import(self):
        try:
            from detector.ai_models.helmet_detector import HelmetDetector
            detector = HelmetDetector()
            self.assertIsNotNone(detector)
        except ImportError as e:
            self.fail(f"Could not import HelmetDetector: {e}")
    
    def test_plate_detector_import(self):
        try:
            from detector.ai_models.plate_detector import LicensePlateDetector
            detector = LicensePlateDetector()
            self.assertIsNotNone(detector)
        except ImportError as e:
            self.fail(f"Could not import LicensePlateDetector: {e}")