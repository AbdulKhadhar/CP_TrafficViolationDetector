import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import django
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'traffic_system.settings')
django.setup()

from detector.models import Violation, Camera, DetectionLog
from detector.ai_models.helmet_detector import HelmetDetector
from detector.ai_models.plate_detector import LicensePlateDetector
from django.core.files.base import ContentFile

class ViolationProcessor:
    """Main processor that combines all detection modules"""
    
    def __init__(self, camera_id, confidence_threshold=0.5):
        """
        Initialize the violation processor
        
        Args:
            camera_id: Database ID of the camera
            confidence_threshold: Minimum confidence for violations
        """
        self.camera_id = camera_id
        self.camera = Camera.objects.get(id=camera_id)
        self.confidence_threshold = confidence_threshold
        
        # Initialize detectors
        self.helmet_detector = HelmetDetector(confidence_threshold=confidence_threshold)
        self.plate_detector = LicensePlateDetector(confidence_threshold=confidence_threshold)
        
        # Create detection log
        self.log = DetectionLog.objects.create(
            camera=self.camera,
            is_running=True
        )
        
        # Rate limiting
        self.last_violation_time = {}
        self.min_time_between_violations = timedelta(seconds=10)
    
    def process_frame(self, frame):
        """
        Process a single frame and detect all violations
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            violations_saved: Number of violations saved to database
            annotated_frame: Frame with annotations
        """
        violations_saved = 0
        annotated_frame = frame.copy()
        
        # Detect helmet violations
        helmet_violations, helmet_annotated = self.helmet_detector.detect_helmet_violation(frame)
        
        # Detect plate violations
        plate_violations, plate_annotated = self.plate_detector.detect_plate_violation(frame)
        
        # Combine annotations
        annotated_frame = cv2.addWeighted(helmet_annotated, 0.5, plate_annotated, 0.5, 0)
        
        # Process helmet violations
        for violation in helmet_violations:
            if self._should_save_violation('helmet', violation['bbox']):
                self._save_violation(
                    frame=frame,
                    violation_type='NO_HELMET',
                    bbox=violation['bbox'],
                    confidence=violation['confidence'],
                    vehicle_bbox=violation.get('vehicle_bbox')
                )
                violations_saved += 1
        
        # Process plate violations
        for violation in plate_violations:
            if self._should_save_violation('plate', violation['bbox']):
                self._save_violation(
                    frame=frame,
                    violation_type='NO_PLATE',
                    bbox=violation['bbox'],
                    confidence=violation['confidence'],
                    vehicle_type=violation['vehicle_type']
                )
                violations_saved += 1
        
        # Update log
        self.log.frames_processed += 1
        self.log.violations_detected += violations_saved
        self.log.save()
        
        return violations_saved, annotated_frame
    
    def _should_save_violation(self, violation_key, bbox):
        """
        Check if violation should be saved (rate limiting)
        
        Args:
            violation_key: Type of violation
            bbox: Bounding box tuple
            
        Returns:
            bool: True if should save
        """
        current_time = datetime.now()
        key = f"{violation_key}_{bbox[0]}_{bbox[1]}"
        
        if key in self.last_violation_time:
            time_diff = current_time - self.last_violation_time[key]
            if time_diff < self.min_time_between_violations:
                return False
        
        self.last_violation_time[key] = current_time
        return True
    
    def _save_violation(self, frame, violation_type, bbox, confidence, 
                       vehicle_bbox=None, vehicle_type='OTHER'):
        """
        Save violation to database
        
        Args:
            frame: Original frame
            violation_type: Type of violation
            bbox: Bounding box of violation
            confidence: Detection confidence
            vehicle_bbox: Bounding box of vehicle (if available)
            vehicle_type: Type of vehicle
        """
        x1, y1, x2, y2 = bbox
        
        # Crop violation region
        violation_crop = frame[y1:y2, x1:x2]
        
        # Create annotated image
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, violation_type, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Try to extract license plate if vehicle bbox available
        plate_number = None
        plate_confidence = None
        
        if vehicle_bbox and violation_type == 'NO_HELMET':
            plate_number, plate_confidence = self.plate_detector.extract_plate_number(
                frame, vehicle_bbox
            )
        
        # Encode images
        _, img_encoded = cv2.imencode('.jpg', violation_crop)
        _, annotated_encoded = cv2.imencode('.jpg', annotated)
        
        # Create violation record
        violation = Violation(
            camera=self.camera,
            violation_type=violation_type,
            vehicle_type=vehicle_type,
            confidence_score=confidence,
            license_plate=plate_number,
            plate_confidence=plate_confidence,
            status='PENDING'
        )
        
        # Save images
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        violation.image.save(
            f'violation_{timestamp}.jpg',
            ContentFile(img_encoded.tobytes()),
            save=False
        )
        violation.annotated_image.save(
            f'annotated_{timestamp}.jpg',
            ContentFile(annotated_encoded.tobytes()),
            save=False
        )
        
        violation.save()
    
    def process_video_file(self, video_path, skip_frames=5):
        """
        Process an uploaded video file
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
            
        Returns:
            total_violations: Total violations detected
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        total_violations = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % skip_frames != 0:
                continue
            
            violations, _ = self.process_frame(frame)
            total_violations += violations
        
        cap.release()
        
        # Update log
        self.log.is_running = False
        self.log.ended_at = datetime.now()
        self.log.save()
        
        return total_violations
    
    def process_live_stream(self, rtsp_url=None, duration_seconds=None):
        """
        Process live video stream from camera
        
        Args:
            rtsp_url: RTSP stream URL (uses camera's URL if None, or webcam if blank)
            duration_seconds: How long to run (None = indefinite)
            
        Returns:
            total_violations: Total violations detected
        """
        # Determine video source
        if rtsp_url is None:
            rtsp_url = self.camera.rtsp_url
        
        # Open video capture
        if rtsp_url:
            print(f"[PROCESSOR] Connecting to RTSP: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
        else:
            print(f"[PROCESSOR] Opening webcam (index 0)")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            
            # Alternative camera indices if 0 doesn't work
            if not cap.isOpened():
                print(f"[PROCESSOR] Webcam 0 failed, trying index 1")
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        # Set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            error_msg = "Failed to open camera/stream"
            print(f"[PROCESSOR ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        print(f"[PROCESSOR] Camera opened successfully!")
        print(f"[PROCESSOR] Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"[PROCESSOR] FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
        
        start_time = datetime.now()
        frame_count = 0
        total_violations = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print(f"[PROCESSOR] Failed to read frame at count {frame_count}")
                    break
                
                frame_count += 1
                
                # Process every 5th frame for efficiency
                if frame_count % 5 == 0:
                    violations, annotated = self.process_frame(frame)
                    total_violations += violations
                    
                    # Print progress every 30 frames
                    if frame_count % 30 == 0:
                        print(f"[PROCESSOR] Processed {frame_count} frames, {total_violations} violations detected")
                
                # Check duration
                if duration_seconds:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration_seconds:
                        print(f"[PROCESSOR] Duration limit reached: {duration_seconds}s")
                        break
        
        except KeyboardInterrupt:
            print(f"[PROCESSOR] Interrupted by user")
        except Exception as e:
            print(f"[PROCESSOR ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            print(f"[PROCESSOR] Camera released")
            
            # Update log
            self.log.is_running = False
            self.log.ended_at = datetime.now()
            self.log.save()
            
            print(f"[PROCESSOR] Final stats: {frame_count} frames, {total_violations} violations")
        
        return total_violations